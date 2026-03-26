"""add root_cause_diagnostic_accuracy to run_metrics

Revision ID: 002_add_diagnostic_accuracy
Revises: 001_initial_schema
Create Date: 2026-03-26

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "002_add_diagnostic_accuracy"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "run_metrics",
        sa.Column("root_cause_diagnostic_accuracy", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("run_metrics", "root_cause_diagnostic_accuracy")
