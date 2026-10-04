"""add experiment document scope and run evaluation/diagnosis timestamps

Revision ID: 006
Revises: 005
Create Date: 2024-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "006_add_experiment_lifecycle"
down_revision = "005_add_eval_sets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Null = search every document (behaviour of experiments created before 006).
    op.add_column("experiments", sa.Column("document_ids", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("evaluated_at", sa.DateTime(), nullable=True))
    op.add_column("runs", sa.Column("diagnosed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "diagnosed_at")
    op.drop_column("runs", "evaluated_at")
    op.drop_column("experiments", "document_ids")
