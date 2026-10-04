"""
Diagnostics endpoints.

Exposes root-cause failure classification results per run and per query,
and triggers XGBoost classifier training.
"""

import logging

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from app.db.session import get_db
from app.db.models import QueryResult, Run, Experiment
from app.services.diagnostics.ml_classifier import is_trained
from app.services.evaluation.eval_sets import EvalSetError, resolve_eval_set_path

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/run/{run_id}")
async def get_run_diagnostics(run_id: str, db: AsyncSession = Depends(get_db)):
    """Return failure category breakdown for all queries in a run."""
    result = await db.execute(
        select(QueryResult)
        .where(QueryResult.run_id == run_id)
        .order_by(QueryResult.query_id)
    )
    query_results = result.scalars().all()
    if not query_results:
        raise HTTPException(status_code=404, detail="No query results found for this run")

    return [
        {
            "query_id": qr.query_id,
            "question": qr.question,
            "generated_answer": qr.generated_answer,
            "failure_category": qr.failure_category,
            "confidence": qr.diagnosis_evidence.get("confidence", 0.7) if qr.diagnosis_evidence else 0.7,
            "evidence": qr.diagnosis_evidence or {},
            "reasoning": (qr.diagnosis_evidence or {}).get("reasoning"),
        }
        for qr in query_results
    ]


@router.get("/run/{run_id}/query/{query_id}")
async def get_query_diagnosis(run_id: str, query_id: str, db: AsyncSession = Depends(get_db)):
    """Return root-cause diagnosis for a single query failure."""
    result = await db.execute(
        select(QueryResult).where(
            QueryResult.run_id == run_id,
            QueryResult.query_id == query_id,
        )
    )
    qr = result.scalar_one_or_none()
    if not qr:
        raise HTTPException(status_code=404, detail="Query result not found")

    evidence = qr.diagnosis_evidence or {}
    return {
        "query_id": qr.query_id,
        "question": qr.question,
        "generated_answer": qr.generated_answer,
        "retrieved_chunks": qr.retrieved_chunks,
        "failure_category": qr.failure_category,
        "confidence": evidence.get("confidence", 0.7),
        "evidence": evidence,
        "reasoning": evidence.get("reasoning"),   # populated after /reason is called
        "latency_ms": qr.latency_ms,
        "cost_usd": qr.cost_usd,
    }


@router.post("/run/{run_id}/query/{query_id}/reason")
async def reason_query_failure(run_id: str, query_id: str, db: AsyncSession = Depends(get_db)):
    """
    Use GPT-4o to reason step-by-step about why a specific query failed.

    Results are cached in diagnosis_evidence["reasoning"] — calling this
    endpoint a second time returns the cached result without a new LLM call.
    Results the model returned unparseable (parse_error) are not cached.
    """
    result = await db.execute(
        select(QueryResult)
        .where(
            QueryResult.run_id == run_id,
            QueryResult.query_id == query_id,
        )
        .order_by(QueryResult.id)
    )
    rows = result.scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="Query result not found")
    if len(rows) > 1:
        logger.warning(
            "Run %s has %d query results with query_id %s; reasoning about the first non-errored one",
            run_id, len(rows), query_id,
        )
    qr = next((r for r in rows if r.status != "error"), rows[0])
    if qr.status == "error":
        raise HTTPException(
            status_code=422,
            detail="Cannot reason about a query that errored (no answer was generated)",
        )

    evidence = dict(qr.diagnosis_evidence or {})

    # Return cached result if already reasoned (never a cached parse failure)
    cached = evidence.get("reasoning")
    if isinstance(cached, dict) and cached and not cached.get("parse_error"):
        return {"query_id": qr.query_id, "cached": True, "reasoning": cached}

    # Build metrics dict from evidence
    metrics = {
        k: evidence[k]
        for k in ("faithfulness", "context_recall", "context_precision", "answer_correctness")
        if k in evidence
    }

    from app.services.diagnostics.reasoning_agent import reason as llm_reason
    try:
        reasoning = await llm_reason(
            question=qr.question,
            generated_answer=qr.generated_answer,
            retrieved_chunks=qr.retrieved_chunks or [],
            metrics=metrics,
            # NULL means not diagnosed yet, not "no failure".
            failure_category=qr.failure_category or "UNDIAGNOSED",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM reasoning failed: {exc}")

    # Cache in DB, unless the model's output could not be parsed (retry next time)
    if not reasoning.get("parse_error"):
        evidence["reasoning"] = reasoning
        qr.diagnosis_evidence = evidence
        await db.commit()

    return {"query_id": qr.query_id, "cached": False, "reasoning": reasoning}


@router.get("/summary")
async def get_failure_summary(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """Aggregate failure categories across all runs in an experiment."""
    # Verify experiment exists
    exp_result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    exp = exp_result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Get all run IDs for this experiment
    runs_result = await db.execute(select(Run.id).where(Run.experiment_id == experiment_id))
    run_ids = [r[0] for r in runs_result.all()]

    if not run_ids:
        return {"failure_counts": {}, "total_queries": 0}

    # Errored queries were never diagnosed; keep them out of the failure counts.
    is_errored = QueryResult.status == "error"

    # Count failure categories
    counts_result = await db.execute(
        select(QueryResult.failure_category, func.count(QueryResult.id).label("count"))
        .where(QueryResult.run_id.in_(run_ids), or_(QueryResult.status.is_(None), ~is_errored))
        .group_by(QueryResult.failure_category)
    )
    # A NULL category on a non-errored row means it was not diagnosed (metrics
    # not scored, or diagnosis not run yet): report it apart, not as NO_FAILURE.
    counts = {}
    undiagnosed = 0
    for category, count in counts_result.all():
        if category is None:
            undiagnosed += count
        else:
            counts[category] = count

    # Total queries
    total_result = await db.execute(
        select(func.count(QueryResult.id)).where(QueryResult.run_id.in_(run_ids))
    )
    total = total_result.scalar() or 0

    errored_result = await db.execute(
        select(func.count(QueryResult.id)).where(QueryResult.run_id.in_(run_ids), is_errored)
    )
    errored = errored_result.scalar() or 0

    return {
        "experiment_id": experiment_id,
        "failure_counts": counts,
        "total_queries": total,
        "errored_queries": errored,
        "undiagnosed_queries": undiagnosed,
        "run_count": len(run_ids),
    }


@router.get("/classifier/status")
async def get_classifier_status():
    """Return whether the ML failure classifier has been trained."""
    return {
        "ml_classifier_trained": is_trained(),
        "classifier_type": "xgboost" if is_trained() else "heuristic",
    }


@router.post("/classifier/train")
async def train_classifier(
    experiment_id: str,
    eval_set_path: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger async XGBoost classifier training on all labeled query results
    from a completed experiment.

    The model trains on failure_category labels written by the heuristic
    classifier (or ground-truth labels in the eval set). Eval items come from
    the experiment's own eval set unless eval_set_path is given. After training,
    all subsequent diagnose() calls will use the ML model automatically.
    """
    # Verify experiment exists
    exp_result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    exp = exp_result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Default: the experiment's own eval set (uploaded, stored path, or
    # DEFAULT_EVAL_SET_PATH). An explicit path must live inside eval_sets/.
    if eval_set_path:
        try:
            eval_set_path, _ = resolve_eval_set_path(eval_set_path)
        except EvalSetError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    from app.workers.tasks import train_failure_classifier
    task = train_failure_classifier.delay(experiment_id, eval_set_path)

    return {
        "task_id": task.id,
        "status": "queued",
        "message": (
            "XGBoost classifier training started. "
            "Call GET /diagnostics/classifier/status to check if training completed."
        ),
    }
