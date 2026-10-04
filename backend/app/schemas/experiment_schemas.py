"""Pydantic schemas for experiment and evaluation endpoints."""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator

from app.services.evaluation.eval_sets import DEFAULT_EVAL_SET_PATH


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
    root_cause_diagnostic_accuracy: Optional[float] = None

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
    status: Optional[str] = "ok"
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class RunResponse(BaseModel):
    id: str
    experiment_id: str
    config: dict
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    diagnosed_at: Optional[datetime] = None
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
    eval_set_id: Optional[str] = None
    eval_set_path: Optional[str] = None
    # Documents retrieval is limited to; null = every document.
    document_ids: Optional[list[str]] = None
    runs: list[RunResponse] = []

    class Config:
        from_attributes = True


class ExperimentCreateRequest(BaseModel):
    name: str
    description: str = ""
    # Preferred: an uploaded eval set id (or "builtin:<file>.json").
    eval_set_id: Optional[str] = None
    # Legacy: a file inside the repo eval_sets/ folder. Ignored when eval_set_id is set.
    eval_set_path: Optional[str] = DEFAULT_EVAL_SET_PATH
    config_preset: str = "mvp"  # mvp | custom
    # Documents retrieval may search; null (default) = every document.
    document_ids: Optional[list[str]] = None

    @field_validator("document_ids")
    @classmethod
    def _non_empty_unique(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        if not v:
            raise ValueError("document_ids must not be empty; use null to search every document")
        return list(dict.fromkeys(v))  # de-duplicate, keep order


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
