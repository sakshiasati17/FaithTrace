"""
Autonomous optimizer agent endpoints.

Create, monitor, and inspect optimizer jobs that iteratively search
for the best RAG pipeline configuration.
"""

import uuid
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import OptimizerJob
from app.schemas.optimizer_schemas import OptimizerCreateRequest, OptimizerJobResponse

router = APIRouter()


@router.post("/", response_model=OptimizerJobResponse, status_code=status.HTTP_201_CREATED)
async def create_optimizer_job(
    payload: OptimizerCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create and start an autonomous optimizer run."""
    from app.services.experiment.optimizer import OptimizerGoal, OptimizerState

    job_id = str(uuid.uuid4())
    goal = OptimizerGoal(
        target_metric=payload.target_metric,
        target_threshold=payload.target_threshold,
        max_iterations=payload.max_iterations,
        max_cost_usd=payload.max_cost_usd,
        eval_set_path=payload.eval_set_path,
    )
    initial_state = OptimizerState(status="pending")

    job = OptimizerJob(
        id=job_id,
        name=payload.name,
        goal=asdict(goal),
        state=asdict(initial_state),
        status="pending",
    )
    db.add(job)
    await db.flush()

    # Enqueue the optimizer agent task
    from app.workers.tasks import run_optimizer_agent
    run_optimizer_agent.delay(job_id)

    return OptimizerJobResponse.model_validate(job)


@router.get("/", response_model=list[OptimizerJobResponse])
async def list_optimizer_jobs(db: AsyncSession = Depends(get_db)):
    """List all optimizer jobs."""
    result = await db.execute(
        select(OptimizerJob).order_by(OptimizerJob.created_at.desc())
    )
    jobs = result.scalars().all()
    return [OptimizerJobResponse.model_validate(j) for j in jobs]


@router.get("/{job_id}", response_model=OptimizerJobResponse)
async def get_optimizer_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get optimizer job status and full iteration history."""
    job = await db.get(OptimizerJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Optimizer job not found")
    return OptimizerJobResponse.model_validate(job)
