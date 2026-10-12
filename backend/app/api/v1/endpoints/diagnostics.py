"""
Diagnostics endpoints.

Exposes root-cause failure classification results per run and per query,
triggers XGBoost classifier training, and re-runs diagnosis (only) for runs
that were already evaluated.
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


# ─── Re-diagnosis (no re-scoring) ────────────────────────────────────────────

def _rediagnose_skip_reason(run: Run) -> str | None:
    """Why a run cannot be re-diagnosed, or None when it can."""
    if run.status != "done":
        return f"run status is '{run.status}', must be 'done'"
    if run.evaluated_at is None:
        return "run has not been evaluated (evaluated_at is not set); evaluate it first"
    return None


async def _score_coverage(db: AsyncSession, run_ids: list[str]) -> dict[str, dict]:
    """
    Per run: total query results, ok (non-errored) queries, and ok queries
    without stored per-query Ragas scores (diagnosis_evidence["scores"]).
    """
    coverage = {rid: {"queries": 0, "ok_queries": 0, "queries_missing_scores": 0} for rid in run_ids}
    if not run_ids:
        return coverage
    rows = await db.execute(
        select(QueryResult.run_id, QueryResult.status, QueryResult.diagnosis_evidence)
        .where(QueryResult.run_id.in_(run_ids))
    )
    for run_id, status, evidence in rows.all():
        entry = coverage.get(run_id)
        if entry is None:
            continue
        entry["queries"] += 1
        if status == "error":
            continue
        entry["ok_queries"] += 1
        if not isinstance((evidence or {}).get("scores"), dict):
            entry["queries_missing_scores"] += 1
    return coverage


def _rescore_counts(entry: dict) -> dict:
    """
    diagnose_run reuses stored scores only when every ok query has them;
    otherwise it re-scores all of the run's ok queries with Ragas (OpenAI cost).
    """
    missing = entry["queries_missing_scores"]
    return {
        "ok_queries": entry["ok_queries"],
        "queries_missing_scores": missing,
        "will_rescore_queries": entry["ok_queries"] if missing else 0,
    }


def _experiment_eval_set(experiment: Experiment | None) -> tuple[str | None, str | None]:
    # Same resolution as POST /evaluation/run/{run_id}: the worker loads the
    # uploaded set (eval_set_id) or the stored built-in path.
    if experiment is None:
        return None, None
    return experiment.eval_set_id, experiment.eval_set_path


@router.post("/run/{run_id}/rediagnose")
async def rediagnose_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """
    Re-run only diagnosis for an evaluated run (e.g. after a classifier change).

    Unlike POST /evaluation/run/{run_id}, this does not re-score the run:
    diagnose_run reuses the per-query Ragas scores evaluate_run stored. If any
    ok query lacks stored scores, diagnose_run re-scores the run's ok queries;
    will_rescore_queries reports how many (0 means no scoring cost).
    """
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    reason = _rediagnose_skip_reason(run)
    if reason:
        raise HTTPException(status_code=400, detail=reason)

    coverage = (await _score_coverage(db, [run_id]))[run_id]
    if coverage["queries"] == 0:
        raise HTTPException(status_code=400, detail="run has no query results to diagnose")

    experiment = await db.get(Experiment, run.experiment_id)
    eval_set_id, eval_set_path = _experiment_eval_set(experiment)

    from app.workers.tasks import diagnose_run
    try:
        diagnose_run.delay(run_id, eval_set_path)
    except Exception as exc:
        logger.exception("Could not enqueue diagnosis for run %s", run_id)
        raise HTTPException(status_code=503, detail=f"Could not enqueue diagnosis: {exc}")

    counts = _rescore_counts(coverage)
    if counts["will_rescore_queries"]:
        logger.warning(
            "Re-diagnosing run %s: %d ok queries lack stored scores; %d will be re-scored",
            run_id, counts["queries_missing_scores"], counts["will_rescore_queries"],
        )
    return {
        "run_id": run_id,
        "status": "enqueued",
        "message": f"Diagnosis enqueued for run {run_id}",
        "eval_set_id": eval_set_id,
        "eval_set_path": eval_set_path,
        **counts,
    }


@router.post("/experiment/{experiment_id}/rediagnose")
async def rediagnose_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """
    Re-run only diagnosis for every eligible run of an experiment (status
    "done" and evaluated). Ineligible runs are reported as skipped with the
    reason; a run whose task could not be enqueued is reported as "error".
    """
    experiment = await db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    eval_set_id, eval_set_path = _experiment_eval_set(experiment)

    runs_result = await db.execute(
        select(Run).where(Run.experiment_id == experiment_id).order_by(Run.created_at, Run.id)
    )
    runs = runs_result.scalars().all()

    reasons = {run.id: _rediagnose_skip_reason(run) for run in runs}
    coverage = await _score_coverage(db, [run.id for run in runs if reasons[run.id] is None])

    from app.workers.tasks import diagnose_run

    per_run = []
    for run in runs:
        entry = {"run_id": run.id, "run_status": run.status}
        reason = reasons[run.id]
        if reason is None and coverage[run.id]["queries"] == 0:
            reason = "run has no query results to diagnose"
        if reason:
            per_run.append({**entry, "status": "skipped", "reason": reason})
            continue
        counts = _rescore_counts(coverage[run.id])
        try:
            diagnose_run.delay(run.id, eval_set_path)
        except Exception as exc:
            logger.exception("Could not enqueue diagnosis for run %s", run.id)
            per_run.append({**entry, "status": "error", "reason": f"could not enqueue diagnosis: {exc}"})
            continue
        per_run.append({**entry, "status": "enqueued", **counts})

    will_rescore = sum(r.get("will_rescore_queries", 0) for r in per_run)
    if will_rescore:
        logger.warning(
            "Re-diagnosing experiment %s: %d ok queries will be re-scored (missing stored scores)",
            experiment_id, will_rescore,
        )
    return {
        "experiment_id": experiment_id,
        "eval_set_id": eval_set_id,
        "eval_set_path": eval_set_path,
        "enqueued": sum(1 for r in per_run if r["status"] == "enqueued"),
        "skipped": sum(1 for r in per_run if r["status"] == "skipped"),
        "errors": sum(1 for r in per_run if r["status"] == "error"),
        "will_rescore_queries": will_rescore,
        "runs": per_run,
    }
