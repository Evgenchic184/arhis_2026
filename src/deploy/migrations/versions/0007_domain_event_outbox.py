"""Add domain event outbox table.

Revision ID: 0007_domain_event_outbox
Revises: 0006_recount_post_comments_total
Create Date: 2026-05-14 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0007_domain_event_outbox"
down_revision = "0006_recount_post_comments_total"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "domain_event_outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("aggregate_type", sa.String(length=64), nullable=False),
        sa.Column("aggregate_id", sa.String(length=64), nullable=True),
        sa.Column("actor_id", sa.String(length=64), nullable=True),
        sa.Column("actor_role", sa.String(length=32), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("topic", sa.String(length=128), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_domain_event_outbox_event_type", "domain_event_outbox", ["event_type"])
    op.create_index("ix_domain_event_outbox_aggregate_type", "domain_event_outbox", ["aggregate_type"])
    op.create_index("ix_domain_event_outbox_aggregate_id", "domain_event_outbox", ["aggregate_id"])
    op.create_index("ix_domain_event_outbox_actor_id", "domain_event_outbox", ["actor_id"])
    op.create_index("ix_domain_event_outbox_request_id", "domain_event_outbox", ["request_id"])
    op.create_index("ix_domain_event_outbox_published_at", "domain_event_outbox", ["published_at"])


def downgrade() -> None:
    op.drop_index("ix_domain_event_outbox_published_at", table_name="domain_event_outbox")
    op.drop_index("ix_domain_event_outbox_request_id", table_name="domain_event_outbox")
    op.drop_index("ix_domain_event_outbox_actor_id", table_name="domain_event_outbox")
    op.drop_index("ix_domain_event_outbox_aggregate_id", table_name="domain_event_outbox")
    op.drop_index("ix_domain_event_outbox_aggregate_type", table_name="domain_event_outbox")
    op.drop_index("ix_domain_event_outbox_event_type", table_name="domain_event_outbox")
    op.drop_table("domain_event_outbox")
