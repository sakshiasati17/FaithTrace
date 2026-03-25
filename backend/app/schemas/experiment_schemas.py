"""Pydantic schemas for experiment and evaluation endpoints."""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class RunMetricsResponse(BaseModel):
    id: str
    run_id: str
    answer_correctness: Optional[float] = None
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevance: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    avg_token_usage: Optional[float] = None
    avg_cost_usd: Optional[float] = None
    freshness_validity: Optional[float] = None
    temporal_citation_accuracy: Optional[float] = None
    multimodal_grounding_rate: Optional[float] = None

    class Config:
        from_attributes = True


class QueryResultResponse(BaseModel):
    id: str
    run_id: str
    query_id: str
    question: str
    generated_answer: str
    retrieved_chunks: list
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    failure_category: Optional[str] = None
    diagnosis_evidence: dict = Field(default_factory=dict)

    class Config:
        from_attributes = True


class RunResponse(BaseModel):
    id: str
    experiment_id: str
    config: dict
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    metrics: Optional[RunMetricsResponse] = None

    class Config:
        from_attributes = True


class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    runs: list[RunResponse] = []

    class Config:
        from_attributes = True


class ExperimentCreateRequest(BaseModel):
    name: str
    description: str = ""
    eval_set_path: str = "eval_sets/sample_eval_set.json"
    config_preset: str = "mvp"  # mvp | custom


class RecommendationResponse(BaseModel):
    objective: str
    best_config: dict
    run_id: str
    score: float
    rationale: str


class LeaderboardEntry(BaseModel):
    run_id: str
    experiment_id: str
    config: dict
    metrics: Optional[RunMetricsResponse] = None
