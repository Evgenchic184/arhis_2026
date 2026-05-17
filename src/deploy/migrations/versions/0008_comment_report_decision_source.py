"""Add decision source and ML score timestamp to comment reports.

Revision ID: 0008_comment_report_ds
Revises: 0007_domain_event_outbox
Create Date: 2026-05-17 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0008_comment_report_ds"
down_revision = "0007_domain_event_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "comment_reports",
        sa.Column("decision_source", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "comment_reports",
        sa.Column("ml_scored_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f("ix_comment_reports_decision_source"),
        "comment_reports",
        ["decision_source"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_comment_reports_decision_source"), table_name="comment_reports")
    op.drop_column("comment_reports", "ml_scored_at")
    op.drop_column("comment_reports", "decision_source")
