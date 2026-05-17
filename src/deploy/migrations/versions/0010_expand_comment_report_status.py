from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0010_expand_cr_status"
down_revision = "0009_model_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "comment_reports",
        "status",
        existing_type=sa.String(length=12),
        type_=sa.String(length=32),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "comment_reports",
        "status",
        existing_type=sa.String(length=32),
        type_=sa.String(length=12),
        existing_nullable=False,
    )
