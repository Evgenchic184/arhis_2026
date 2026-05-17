"""Add ML model stage to comment reports.

Revision ID: 0012_comment_report_ml_stage
Revises: 0011_system_alerts
Create Date: 2026-05-17 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0012_comment_report_ml_stage"
down_revision = "0011_system_alerts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("comment_reports", sa.Column("ml_model_stage", sa.String(length=16), nullable=True))
    op.create_index(op.f("ix_comment_reports_ml_model_stage"), "comment_reports", ["ml_model_stage"])


def downgrade() -> None:
    op.drop_index(op.f("ix_comment_reports_ml_model_stage"), table_name="comment_reports")
    op.drop_column("comment_reports", "ml_model_stage")
