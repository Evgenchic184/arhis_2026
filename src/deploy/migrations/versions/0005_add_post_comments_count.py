"""Add comments_count to posts.

Revision ID: 0005_add_post_comments_count
Revises: 0004_cleanup_user_state_columns
Create Date: 2026-05-13 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision = "0005_add_post_comments_count"
down_revision = "0004_cleanup_user_state_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "posts",
        sa.Column(
            "comments_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )

    op.execute(
        text(
            """
            UPDATE posts
            SET comments_count = COALESCE((
                SELECT COUNT(*)
                FROM comments
                WHERE comments.post_id = posts.id
                  AND comments.visibility = 'visible'
            ), 0)
            """
        )
    )

    op.alter_column("posts", "comments_count", server_default=None, existing_type=sa.Integer(), existing_nullable=False)


def downgrade() -> None:
    op.drop_column("posts", "comments_count")
