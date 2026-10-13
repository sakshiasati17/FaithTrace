"""add temporal-family diagnostic accuracy to run metrics

Revision ID: 007
Revises: 006
Create Date: 2026-10-13 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "007_add_temporal_family_accuracy"
down_revision = "006_add_experiment_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Null until diagnose_run computes it (and when no item is labelled).
    op.add_column(
        "run_metrics",
        sa.Column("temporal_family_diagnostic_accuracy", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run_metrics", "temporal_family_diagnostic_accuracy")
