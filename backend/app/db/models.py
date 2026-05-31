"""
SQLAlchemy ORM models.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)  # pdf, docx, xlsx, csv, html
    version_label: Mapped[str] = mapped_column(String, default="v1")
    effective_from: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    effective_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    parse_status: Mapped[str] = mapped_column(String, default="pending")  # pending, running, done, failed
    index_status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String, default="pending")  # pending, running, done, failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    runs: Mapped[list["Run"]] = relationship("Run", back_populates="experiment")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(String, ForeignKey("experiments.id"), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)   # PipelineConfig as dict
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="runs")
    metrics: Mapped["RunMetrics"] = relationship("RunMetrics", back_populates="run", uselist=False)
    query_results: Mapped[list["QueryResult"]] = relationship("QueryResult", back_populates="run")


class RunMetrics(Base):
    __tablename__ = "run_metrics"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"), nullable=False)

    # Standard RAG metrics
    answer_correctness: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Operational metrics
    latency_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_token_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # FaithTrace custom metrics
    freshness_validity: Mapped[float | None] = mapped_column(Float, nullable=True)
    temporal_citation_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    multimodal_grounding_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    root_cause_diagnostic_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped["Run"] = relationship("Run", back_populates="metrics")


class QueryResult(Base):
    __tablename__ = "query_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"), nullable=False)
    query_id: Mapped[str] = mapped_column(String, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    generated_answer: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunks: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    failure_category: Mapped[str | None] = mapped_column(String, nullable=True)
    diagnosis_evidence: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped["Run"] = relationship("Run", back_populates="query_results")
    feedback: Mapped[list["QueryFeedback"]] = relationship("QueryFeedback", back_populates="query_result")


class QueryFeedback(Base):
    __tablename__ = "query_feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_result_id: Mapped[str] = mapped_column(String, ForeignKey("query_results.id"), nullable=False)
    rating: Mapped[str] = mapped_column(String, nullable=False)   # "positive" | "negative"
    correct_label: Mapped[str | None] = mapped_column(String, nullable=True)  # optional override failure category
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    query_result: Mapped["QueryResult"] = relationship("QueryResult", back_populates="feedback")


class OptimizerJob(Base):
    __tablename__ = "optimizer_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False, default="Auto-Optimizer")
    goal: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending, running, converged, budget_exceeded, max_iterations, failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
