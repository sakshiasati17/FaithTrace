"""
Diagnostics endpoints.

Exposes root-cause failure classification results per run and per query.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/run/{run_id}")
async def get_run_diagnostics(run_id: str):
    """Return failure category breakdown for all queries in a run."""
    raise NotImplementedError


@router.get("/run/{run_id}/query/{query_id}")
async def get_query_diagnosis(run_id: str, query_id: str):
    """Return root-cause diagnosis for a single query failure."""
    raise NotImplementedError


@router.get("/summary")
async def get_failure_summary(experiment_id: str):
    """Aggregate failure categories across all runs in an experiment."""
    raise NotImplementedError
