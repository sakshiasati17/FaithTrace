"""
Diagnostics endpoints.

Exposes root-cause failure classification results per run and per query.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.session import get_db
from app.db.models import QueryResult, Run, Experiment

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

    return {
        "query_id": qr.query_id,
        "question": qr.question,
        "generated_answer": qr.generated_answer,
        "retrieved_chunks": qr.retrieved_chunks,
        "failure_category": qr.failure_category,
        "confidence": qr.diagnosis_evidence.get("confidence", 0.7) if qr.diagnosis_evidence else 0.7,
        "evidence": qr.diagnosis_evidence or {},
        "latency_ms": qr.latency_ms,
        "cost_usd": qr.cost_usd,
    }


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

    # Count failure categories
    counts_result = await db.execute(
        select(QueryResult.failure_category, func.count(QueryResult.id).label("count"))
        .where(QueryResult.run_id.in_(run_ids))
        .group_by(QueryResult.failure_category)
    )
    counts = {row[0] or "NO_FAILURE": row[1] for row in counts_result.all()}

    # Total queries
    total_result = await db.execute(
        select(func.count(QueryResult.id)).where(QueryResult.run_id.in_(run_ids))
    )
    total = total_result.scalar() or 0

    return {
        "experiment_id": experiment_id,
        "failure_counts": counts,
        "total_queries": total,
        "run_count": len(run_ids),
    }
