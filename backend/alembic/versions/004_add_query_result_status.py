"""add status and error_message to query_results

Revision ID: 004
Revises: 003
Create Date: 2024-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "004_add_query_result_status"
down_revision = "003_add_query_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "query_results",
        sa.Column("status", sa.String(), nullable=True, server_default="ok"),
    )
    op.add_column(
        "query_results",
        sa.Column("error_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("query_results", "error_message")
    op.drop_column("query_results", "status")
