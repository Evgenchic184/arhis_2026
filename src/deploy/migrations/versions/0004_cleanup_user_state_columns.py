"""Cleanup old user state columns and tables.

Revision ID: 0004_cleanup_user_state_columns
Revises: 0003_add_simple_user_counters
Create Date: 2026-05-13 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text


revision = "0004_cleanup_user_state_columns"
down_revision = "0003_add_simple_user_counters"
branch_labels = None
depends_on = None


OLD_USER_COLUMNS = [
    "account_age_days",
    "reputation_score",
    "reports_last_24h",
    "reports_last_7d",
    "reports_last_30d",
    "deleted_comments_last_1d",
    "deleted_comments_last_7d",
    "deleted_comments_last_30d",
    "hidden_comments_last_1d",
    "hidden_comments_last_7d",
    "hidden_comments_last_30d",
    "comment_count_last_1d",
    "comment_count_last_7d",
    "comment_count_last_30d",
    "auto_action_count_last_30d",
    "manual_overrule_count_last_30d",
    "auto_action_rate_last_30d",
    "manual_overrule_rate_last_30d",
    "last_ml_confidence",
    "last_ml_verdict",
    "last_ml_model_version",
]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    user_columns = {column["name"] for column in inspector.get_columns("users")}

    for column_name in OLD_USER_COLUMNS:
        if column_name in user_columns:
            op.drop_column("users", column_name)

    if "user_feature_states" in inspector.get_table_names():
        op.drop_table("user_feature_states")


def downgrade() -> None:
    # This cleanup migration is intentionally one-way.
    pass
