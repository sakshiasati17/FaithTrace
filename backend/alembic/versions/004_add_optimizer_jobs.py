"""Add optimizer_jobs table.

Revision ID: 004_add_optimizer_jobs
Revises: 003_add_query_feedback
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "004_add_optimizer_jobs"
down_revision = "003_add_query_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "optimizer_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, server_default="Auto-Optimizer"),
        sa.Column("goal", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("state", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(), server_default="pending"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("optimizer_jobs")
