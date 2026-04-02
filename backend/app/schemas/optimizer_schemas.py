"""Pydantic schemas for the optimizer endpoints."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class OptimizerCreateRequest(BaseModel):
    name: str = "Auto-Optimizer"
    target_metric: str = "faithfulness"
    target_threshold: float = 0.85
    max_iterations: int = Field(default=5, ge=1, le=10)
    max_cost_usd: float = Field(default=0.50, ge=0.0)
    eval_set_path: str = "eval_sets/sample_eval_set.json"


class OptimizerHistoryEntry(BaseModel):
    iteration: int
    experiment_id: str
    configs_tested: int
    best_score: float
    best_config: Optional[dict] = None


class OptimizerJobResponse(BaseModel):
    id: str
    name: str
    goal: dict
    state: dict
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
