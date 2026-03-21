"""
Experiment orchestration endpoints.

Manages pipeline configuration matrices, run scheduling, and status tracking.
"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def create_experiment():
    """Create and enqueue a new RAG pipeline experiment."""
    raise NotImplementedError


@router.get("/")
async def list_experiments():
    """List all experiments with status and summary metrics."""
    raise NotImplementedError


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get detailed results for a specific experiment run."""
    raise NotImplementedError


@router.get("/{experiment_id}/runs")
async def list_runs(experiment_id: str):
    """List all pipeline runs within an experiment."""
    raise NotImplementedError


@router.get("/{experiment_id}/runs/{run_id}/trace")
async def get_run_trace(experiment_id: str, run_id: str):
    """Get per-query trace with retrieved chunks, citations, and scores."""
    raise NotImplementedError
