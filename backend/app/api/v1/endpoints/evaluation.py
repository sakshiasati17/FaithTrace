"""
Evaluation endpoints.

Triggers metric computation and returns standard + custom metric results.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.db.models import Run, RunMetrics, Experiment
from app.schemas.experiment_schemas import RunMetricsResponse, LeaderboardEntry

router = APIRouter()


@router.post("/run/{run_id}")
async def trigger_evaluate_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """Manually trigger metric computation for a completed pipeline run."""
    result = await db.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "done":
        raise HTTPException(status_code=400, detail=f"Run status is '{run.status}', must be 'done'")

    from app.workers.tasks import evaluate_run
    evaluate_run.delay(run_id)
    return {"message": f"Evaluation enqueued for run {run_id}"}


@router.get("/run/{run_id}/metrics", response_model=RunMetricsResponse)
async def get_run_metrics(run_id: str, db: AsyncSession = Depends(get_db)):
    """Return full metric breakdown for a pipeline run."""
    result = await db.execute(select(RunMetrics).where(RunMetrics.run_id == run_id))
    metrics = result.scalar_one_or_none()
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Metrics not yet computed for this run. Run evaluation first."
        )
    return RunMetricsResponse.model_validate(metrics)


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    experiment_id: str | None = None,
    sort_by: str = "faithfulness",
    db: AsyncSession = Depends(get_db),
):
    """Return ranked pipeline configurations by selected metric."""
    valid_sort_fields = [
        "faithfulness", "answer_correctness", "context_recall", "context_precision",
        "answer_relevance", "latency_p50_ms", "avg_cost_usd", "freshness_validity",
        "multimodal_grounding_rate",
    ]
    if sort_by not in valid_sort_fields:
        sort_by = "faithfulness"

    # Fetch all runs with metrics
    query = (
        select(Run, RunMetrics)
        .join(RunMetrics, Run.id == RunMetrics.run_id)
        .where(Run.status == "done")
    )
    if experiment_id:
        query = query.where(Run.experiment_id == experiment_id)

    result = await db.execute(query)
    rows = result.all()

    entries = []
    for run, metrics in rows:
        entries.append(LeaderboardEntry(
            run_id=run.id,
            experiment_id=run.experiment_id,
            config=run.config,
            metrics=RunMetricsResponse.model_validate(metrics),
        ))

    # Sort: for cost and latency, lower is better; for others, higher is better
    lower_is_better = {"latency_p50_ms", "avg_cost_usd"}
    reverse = sort_by not in lower_is_better

    def sort_key(entry: LeaderboardEntry):
        if entry.metrics:
            val = getattr(entry.metrics, sort_by, None)
            return val if val is not None else (-1 if reverse else 9999)
        return -1 if reverse else 9999

    entries.sort(key=sort_key, reverse=reverse)
    return entries


@router.get("/baseline-comparison")
async def get_baseline_comparison(
    experiment_id: str,
    target_run_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Compare pipeline runs against the naive baseline configuration.

    The baseline is: vector_only retrieval, fixed_size chunking, text_only
    parsing, no freshness filtering. Returns per-metric deltas showing
    absolute and relative improvement.

    If target_run_id is omitted, compares the best overall run to baseline.
    """
    exp_result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    if not exp_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Experiment not found")

    query = (
        select(Run, RunMetrics)
        .join(RunMetrics, Run.id == RunMetrics.run_id)
        .where(Run.experiment_id == experiment_id, Run.status == "done")
    )
    result = await db.execute(query)
    rows = result.all()

    if not rows:
        raise HTTPException(status_code=404, detail="No completed runs with metrics found")

    runs = []
    for run, metrics in rows:
        runs.append({
            "run_id": run.id,
            "config": run.config,
            "metrics": {
                "faithfulness": metrics.faithfulness,
                "answer_correctness": metrics.answer_correctness,
                "context_recall": metrics.context_recall,
                "context_precision": metrics.context_precision,
                "answer_relevance": metrics.answer_relevance,
                "latency_p50_ms": metrics.latency_p50_ms,
                "avg_cost_usd": metrics.avg_cost_usd,
                "freshness_validity": metrics.freshness_validity,
                "multimodal_grounding_rate": metrics.multimodal_grounding_rate,
            },
        })

    from app.services.evaluation.baseline import compare_run_to_baseline
    return compare_run_to_baseline(runs, target_run_id)
