"""add query_feedback table

Revision ID: 003
Revises: 002
Create Date: 2024-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_feedback",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("query_result_id", sa.String(), sa.ForeignKey("query_results.id"), nullable=False),
        sa.Column("rating", sa.String(), nullable=False),
        sa.Column("correct_label", sa.String(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_feedback_query_result_id", "query_feedback", ["query_result_id"])


def downgrade() -> None:
    op.drop_index("ix_query_feedback_query_result_id", table_name="query_feedback")
    op.drop_table("query_feedback")
