"""Add model registry tables.

Revision ID: 0009_model_registry
Revises: 0008_comment_report_ds
Create Date: 2026-05-17 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0009_model_registry"
down_revision = "0008_comment_report_ds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("artifact_uri", sa.String(length=512), nullable=False),
        sa.Column("metadata_uri", sa.String(length=512), nullable=True),
        sa.Column("feature_config_version", sa.Integer(), nullable=False),
        sa.Column("traffic_percent", sa.Integer(), nullable=False),
        sa.Column("required_validation_accuracy", sa.Float(), nullable=False),
        sa.Column("validation_sample_size", sa.Integer(), nullable=False),
        sa.Column("validation_accuracy", sa.Float(), nullable=True),
        sa.Column("validation_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("training_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("active_from_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.UniqueConstraint("model_name", "version", name=op.f("uq_model_versions_model_name_version")),
    )
    op.create_index(op.f("ix_model_versions_model_name"), "model_versions", ["model_name"])
    op.create_index(op.f("ix_model_versions_status"), "model_versions", ["status"])
    op.create_index(op.f("ix_model_versions_version"), "model_versions", ["version"])


def downgrade() -> None:
    op.drop_index(op.f("ix_model_versions_version"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_status"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_model_name"), table_name="model_versions")
    op.drop_table("model_versions")
