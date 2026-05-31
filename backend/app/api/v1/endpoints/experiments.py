"""
Experiment orchestration endpoints.

Manages pipeline configuration matrices, run scheduling, and status tracking.
"""

import uuid
from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.db.models import Experiment, Run, QueryResult as QueryResultModel
from app.schemas.experiment_schemas import (
    ExperimentResponse, ExperimentCreateRequest, RunResponse, QueryResultResponse
)

router = APIRouter()


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    payload: ExperimentCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create and enqueue a new RAG pipeline experiment."""
    from app.services.experiment.config_matrix import build_mvp_matrix, build_matrix

    experiment_id = str(uuid.uuid4())
    experiment = Experiment(
        id=experiment_id,
        name=payload.name,
        description=payload.description,
        status="pending",
    )
    db.add(experiment)
    await db.flush()

    # Build configs
    configs = build_mvp_matrix() if payload.config_preset == "mvp" else build_matrix()

    runs = []
    for config in configs:
        run = Run(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            config=asdict(config),
            status="pending",
        )
        db.add(run)
        runs.append(run)

    # Commit before enqueuing — the Celery worker runs in a separate process and
    # reads from the DB immediately. flush() only makes rows visible within this
    # session; commit() makes them visible to other connections.
    await db.commit()

    # Enqueue the experiment run
    from app.workers.tasks import run_experiment as run_experiment_task
    run_experiment_task.delay(experiment_id, payload.eval_set_path)

    # Eager load runs for response
    result = await db.execute(
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(selectinload(Experiment.runs).selectinload(Run.metrics))
    )
    exp = result.scalar_one()
    return ExperimentResponse.model_validate(exp)


@router.get("/", response_model=list[ExperimentResponse])
async def list_experiments(db: AsyncSession = Depends(get_db)):
    """List all experiments with status and summary metrics."""
    result = await db.execute(
        select(Experiment)
        .options(selectinload(Experiment.runs).selectinload(Run.metrics))
        .order_by(Experiment.created_at.desc())
    )
    experiments = result.scalars().all()
    return [ExperimentResponse.model_validate(e) for e in experiments]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """Get detailed results for a specific experiment run."""
    result = await db.execute(
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(selectinload(Experiment.runs).selectinload(Run.metrics))
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse.model_validate(exp)


@router.get("/{experiment_id}/runs", response_model=list[RunResponse])
async def list_runs(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """List all pipeline runs within an experiment."""
    result = await db.execute(
        select(Run)
        .where(Run.experiment_id == experiment_id)
        .options(selectinload(Run.metrics))
        .order_by(Run.created_at.asc())
    )
    runs = result.scalars().all()
    return [RunResponse.model_validate(r) for r in runs]


@router.get("/{experiment_id}/runs/{run_id}/trace", response_model=list[QueryResultResponse])
async def get_run_trace(experiment_id: str, run_id: str, db: AsyncSession = Depends(get_db)):
    """Get per-query trace with retrieved chunks, citations, and scores."""
    # Verify run belongs to experiment
    result = await db.execute(
        select(Run).where(Run.id == run_id, Run.experiment_id == experiment_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    qr_result = await db.execute(
        select(QueryResultModel)
        .where(QueryResultModel.run_id == run_id)
        .order_by(QueryResultModel.query_id)
    )
    query_results = qr_result.scalars().all()
    return [QueryResultResponse.model_validate(qr) for qr in query_results]
