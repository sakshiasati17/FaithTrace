"""
Recommendation endpoints.

Returns the best pipeline configuration per objective based on experiment results.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_recommendations(experiment_id: str):
    """
    Return configuration recommendations for each objective:
    best_overall, lowest_cost, best_latency, best_for_tables, best_for_drift.
    """
    raise NotImplementedError


@router.get("/explain/{run_id}")
async def explain_recommendation(run_id: str):
    """Return a natural-language explanation for why this configuration was recommended."""
    raise NotImplementedError
