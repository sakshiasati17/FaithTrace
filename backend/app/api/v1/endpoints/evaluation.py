"""
Evaluation endpoints.

Triggers metric computation and returns standard + custom metric results.
"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/run/{run_id}")
async def evaluate_run(run_id: str):
    """Compute all metrics for a completed pipeline run."""
    raise NotImplementedError


@router.get("/run/{run_id}/metrics")
async def get_run_metrics(run_id: str):
    """Return full metric breakdown for a pipeline run."""
    raise NotImplementedError


@router.get("/leaderboard")
async def get_leaderboard(experiment_id: str | None = None):
    """Return ranked pipeline configurations by selected metric."""
    raise NotImplementedError
