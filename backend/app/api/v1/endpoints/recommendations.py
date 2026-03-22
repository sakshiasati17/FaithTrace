"""
Recommendation endpoints.

Returns the best pipeline configuration per objective based on experiment results.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Run, RunMetrics, Experiment, QueryResult
from app.schemas.experiment_schemas import RecommendationResponse

router = APIRouter()


@router.get("/", response_model=list[RecommendationResponse])
async def get_recommendations(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """
    Return configuration recommendations for each objective:
    best_overall, lowest_cost, best_latency, best_for_tables, best_for_drift.
    """
    # Verify experiment exists
    exp_result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    exp = exp_result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Fetch all runs with metrics for this experiment
    result = await db.execute(
        select(Run, RunMetrics)
        .join(RunMetrics, Run.id == RunMetrics.run_id)
        .where(Run.experiment_id == experiment_id, Run.status == "done")
    )
    rows = result.all()

    if not rows:
        return []

    # Build run dicts for the recommendation engine
    runs = []
    for run, metrics in rows:
        runs.append({
            "run_id": run.id,
            "config": run.config,
            "metrics": {
                "faithfulness": metrics.faithfulness,
                "answer_correctness": metrics.answer_correctness,
                "context_precision": metrics.context_precision,
                "context_recall": metrics.context_recall,
                "answer_relevance": metrics.answer_relevance,
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p95_ms": metrics.latency_p95_ms,
                "avg_cost_usd": metrics.avg_cost_usd,
                "freshness_validity": metrics.freshness_validity,
                "multimodal_grounding_rate": metrics.multimodal_grounding_rate,
            },
        })

    from app.services.recommendation.engine import recommend
    recommendations = recommend(runs)

    return [
        RecommendationResponse(
            objective=rec.objective,
            best_config=rec.best_config,
            run_id=rec.run_id,
            score=rec.score,
            rationale=rec.rationale,
        )
        for rec in recommendations
    ]


@router.get("/explain/{run_id}")
async def explain_recommendation(run_id: str, db: AsyncSession = Depends(get_db)):
    """Return a natural-language explanation for why this configuration was recommended."""
    result = await db.execute(
        select(Run, RunMetrics)
        .join(RunMetrics, Run.id == RunMetrics.run_id)
        .where(Run.id == run_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found or metrics not computed")

    run, metrics = row
    config = run.config

    rationale = (
        f"Configuration: retrieval={config.get('retrieval_strategy')}, "
        f"chunking={config.get('chunking_strategy')}, "
        f"parsing={config.get('parsing_strategy')}, "
        f"freshness={config.get('freshness_policy')}. "
        f"Key metrics: faithfulness={metrics.faithfulness:.3f}, "
        f"context_recall={metrics.context_recall:.3f}, "
        f"latency_p50={metrics.latency_p50_ms:.0f}ms, "
        f"cost=${metrics.avg_cost_usd:.4f}/query."
    )

    return {"run_id": run_id, "rationale": rationale, "config": config}
