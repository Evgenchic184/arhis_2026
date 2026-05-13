"""Recount post comments as total thread size.

Revision ID: 0006_recount_post_comments_total
Revises: 0005_add_post_comments_count
Create Date: 2026-05-13 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text


revision = "0006_recount_post_comments_total"
down_revision = "0005_add_post_comments_count"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        text(
            """
            UPDATE posts
            SET comments_count = COALESCE((
                SELECT COUNT(*)
                FROM comments
                WHERE comments.post_id = posts.id
            ), 0)
            """
        )
    )


def downgrade() -> None:
    pass
