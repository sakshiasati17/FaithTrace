"""initial schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-03-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("file_type", sa.String(), nullable=False),
        sa.Column("version_label", sa.String(), server_default="v1"),
        sa.Column("effective_from", sa.DateTime(), nullable=True),
        sa.Column("effective_to", sa.DateTime(), nullable=True),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("parse_status", sa.String(), server_default="pending"),
        sa.Column("index_status", sa.String(), server_default="pending"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("doc_metadata", sa.JSON(), server_default="{}"),
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("status", sa.String(), server_default="pending"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("experiment_id", sa.String(), sa.ForeignKey("experiments.id"), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), server_default="pending"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "run_metrics",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("answer_correctness", sa.Float(), nullable=True),
        sa.Column("faithfulness", sa.Float(), nullable=True),
        sa.Column("context_precision", sa.Float(), nullable=True),
        sa.Column("context_recall", sa.Float(), nullable=True),
        sa.Column("answer_relevance", sa.Float(), nullable=True),
        sa.Column("latency_p50_ms", sa.Float(), nullable=True),
        sa.Column("latency_p95_ms", sa.Float(), nullable=True),
        sa.Column("avg_token_usage", sa.Float(), nullable=True),
        sa.Column("avg_cost_usd", sa.Float(), nullable=True),
        sa.Column("freshness_validity", sa.Float(), nullable=True),
        sa.Column("temporal_citation_accuracy", sa.Float(), nullable=True),
        sa.Column("multimodal_grounding_rate", sa.Float(), nullable=True),
    )

    op.create_table(
        "query_results",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("query_id", sa.String(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("generated_answer", sa.Text(), nullable=False),
        sa.Column("retrieved_chunks", sa.JSON(), server_default="[]"),
        sa.Column("latency_ms", sa.Float(), server_default="0.0"),
        sa.Column("input_tokens", sa.Integer(), server_default="0"),
        sa.Column("output_tokens", sa.Integer(), server_default="0"),
        sa.Column("cost_usd", sa.Float(), server_default="0.0"),
        sa.Column("failure_category", sa.String(), nullable=True),
        sa.Column("diagnosis_evidence", sa.JSON(), server_default="{}"),
    )

    # Indexes
    op.create_index("ix_runs_experiment_id", "runs", ["experiment_id"])
    op.create_index("ix_run_metrics_run_id", "run_metrics", ["run_id"])
    op.create_index("ix_query_results_run_id", "query_results", ["run_id"])
    op.create_index("ix_query_results_query_id", "query_results", ["query_id"])


def downgrade() -> None:
    op.drop_table("query_results")
    op.drop_table("run_metrics")
    op.drop_table("runs")
    op.drop_table("experiments")
    op.drop_table("documents")
