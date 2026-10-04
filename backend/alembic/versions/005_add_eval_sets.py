"""add eval_sets table and experiment eval set columns

Revision ID: 005
Revises: 004
Create Date: 2024-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "005_add_eval_sets"
down_revision = "004_add_query_result_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_sets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True, server_default=""),
        sa.Column("source", sa.String(), nullable=False, server_default="upload"),
        sa.Column("filename", sa.String(), nullable=True),
        sa.Column("items", sa.JSON(), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
    )
    op.add_column(
        "experiments",
        sa.Column("eval_set_id", sa.String(), sa.ForeignKey("eval_sets.id"), nullable=True),
    )
    op.add_column(
        "experiments",
        sa.Column("eval_set_path", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("experiments", "eval_set_path")
    op.drop_column("experiments", "eval_set_id")
    op.drop_table("eval_sets")
