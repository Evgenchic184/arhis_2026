"""Add simple user counters.

Revision ID: 0003_add_simple_user_counters
Revises: 0002_user_auth_and_ml_fields
Create Date: 2026-05-13 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_add_simple_user_counters"
down_revision = "0002_user_auth_and_ml_fields"
branch_labels = None
depends_on = None


COUNTER_COLUMNS = [
    ("posts_count", sa.Integer(), sa.text("0")),
    ("comments_count", sa.Integer(), sa.text("0")),
    ("reports_count", sa.Integer(), sa.text("0")),
    ("deleted_comments_count", sa.Integer(), sa.text("0")),
    ("hidden_comments_count", sa.Integer(), sa.text("0")),
]


def upgrade() -> None:
    for column_name, column_type, default in COUNTER_COLUMNS:
        op.add_column("users", sa.Column(column_name, column_type, nullable=False, server_default=default))


def downgrade() -> None:
    for column_name, _, _ in reversed(COUNTER_COLUMNS):
        op.drop_column("users", column_name)
