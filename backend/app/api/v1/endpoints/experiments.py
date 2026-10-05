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
from app.db.models import Document, EvalSet, Experiment, Run, QueryResult as QueryResultModel
from app.schemas.experiment_schemas import (
    ExperimentResponse, ExperimentCreateRequest, RunResponse, QueryResultResponse
)
from app.services.evaluation import eval_sets as es

router = APIRouter()


async def _resolve_eval_set(
    payload: ExperimentCreateRequest, db: AsyncSession
) -> tuple[str | None, str | None]:
    """
    Pick the experiment's eval set: (eval_set_id, eval_set_path).

    eval_set_id wins; "builtin:<file>" ids map to a path. A legacy path must
    resolve inside the repo eval_sets/ folder, else 422.
    """
    if payload.eval_set_id:
        if payload.eval_set_id.startswith(es.BUILTIN_ID_PREFIX):
            path = payload.eval_set_id[len(es.BUILTIN_ID_PREFIX):]
        else:
            if not await db.get(EvalSet, payload.eval_set_id):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Eval set {payload.eval_set_id!r} not found",
                )
            return payload.eval_set_id, None
    else:
        path = payload.eval_set_path or es.DEFAULT_EVAL_SET_PATH

    try:
        normalised, _ = es.resolve_eval_set_path(path)
    except es.EvalSetError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return None, normalised


async def _validate_document_ids(document_ids: list[str] | None, db: AsyncSession) -> None:
    """422 unless every id names an existing document. None (all documents) is valid."""
    if document_ids is None:
        return
    found = set((await db.execute(
        select(Document.id).where(Document.id.in_(document_ids))
    )).scalars().all())
    missing = [d for d in document_ids if d not in found]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown document id(s): {', '.join(missing)}",
        )


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    payload: ExperimentCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create and enqueue a new RAG pipeline experiment."""
    from app.services.experiment.config_matrix import build_configs, build_mvp_matrix, build_matrix

    eval_set_id, eval_set_path = await _resolve_eval_set(payload, db)
    await _validate_document_ids(payload.document_ids, db)

    experiment_id = str(uuid.uuid4())
    experiment = Experiment(
        id=experiment_id,
        name=payload.name,
        description=payload.description,
        status="pending",
        eval_set_id=eval_set_id,
        eval_set_path=eval_set_path,
        document_ids=payload.document_ids,
    )
    db.add(experiment)
    await db.flush()

    # Build configs: an explicit list wins over the preset.
    if payload.configs is not None:
        configs = build_configs(spec.model_dump() for spec in payload.configs)
    else:
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
    # The worker reads the eval set from the experiment row; the path is passed
    # only for legacy path-based experiments.
    run_experiment_task.delay(experiment_id, eval_set_path)

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
