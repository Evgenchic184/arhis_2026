"""Add system alerts table.

Revision ID: 0011_system_alerts
Revises: 0010_expand_cr_status
Create Date: 2026-05-17 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0011_system_alerts"
down_revision = "0010_expand_cr_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fingerprint", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("alertname", sa.String(length=128), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("labels", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("annotations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("receiver", sa.String(length=128), nullable=True),
        sa.Column("generator_url", sa.Text(), nullable=True),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_system_alerts")),
        sa.UniqueConstraint("fingerprint", name=op.f("uq_system_alerts_fingerprint")),
    )
    op.create_index(op.f("ix_system_alerts_alertname"), "system_alerts", ["alertname"])
    op.create_index(op.f("ix_system_alerts_is_active"), "system_alerts", ["is_active"])
    op.create_index(op.f("ix_system_alerts_received_at"), "system_alerts", ["received_at"])
    op.create_index(op.f("ix_system_alerts_resolved_at"), "system_alerts", ["resolved_at"])
    op.create_index(op.f("ix_system_alerts_severity"), "system_alerts", ["severity"])
    op.create_index(op.f("ix_system_alerts_status"), "system_alerts", ["status"])
    op.create_index(op.f("ix_system_alerts_updated_at"), "system_alerts", ["updated_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_system_alerts_updated_at"), table_name="system_alerts")
    op.drop_index(op.f("ix_system_alerts_status"), table_name="system_alerts")
    op.drop_index(op.f("ix_system_alerts_severity"), table_name="system_alerts")
    op.drop_index(op.f("ix_system_alerts_resolved_at"), table_name="system_alerts")
    op.drop_index(op.f("ix_system_alerts_received_at"), table_name="system_alerts")
    op.drop_index(op.f("ix_system_alerts_is_active"), table_name="system_alerts")
    op.drop_index(op.f("ix_system_alerts_alertname"), table_name="system_alerts")
    op.drop_table("system_alerts")
