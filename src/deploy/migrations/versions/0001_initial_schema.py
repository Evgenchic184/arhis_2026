"""Initial schema for posts, comments, reports, and moderation.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-11 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column(
            "role",
            sa.Enum("user", "moderator", "admin", name="user_role", native_enum=False),
            nullable=False,
            server_default=sa.text("'user'"),
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.UniqueConstraint("username", name="uq_users_username"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "posts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "author_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("is_published", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.create_index("ix_posts_author_id", "posts", ["author_id"], unique=False)

    op.create_table(
        "comments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "post_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("posts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "author_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_comment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("comments.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "visibility",
            sa.Enum("visible", "hidden", "deleted", name="comment_visibility", native_enum=False),
            nullable=False,
            server_default=sa.text("'visible'"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index("ix_comments_post_id", "comments", ["post_id"], unique=False)
    op.create_index("ix_comments_author_id", "comments", ["author_id"], unique=False)
    op.create_index("ix_comments_parent_comment_id", "comments", ["parent_comment_id"], unique=False)

    op.create_table(
        "comment_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "comment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("comments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "reporter_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "reason",
            sa.Enum(
                "harassment",
                "hate_speech",
                "spam",
                "abuse",
                "other",
                name="report_reason",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("reason_text", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "under_review", "resolved", "dismissed", name="report_status", native_enum=False),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "moderation_verdict",
            sa.Enum(
                "toxic",
                "not_toxic",
                "needs_ml_review",
                name="moderation_verdict",
                native_enum=False,
            ),
            nullable=True,
        ),
        sa.Column("moderation_note", sa.Text(), nullable=True),
        sa.Column(
            "reviewed_by_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ml_score", sa.Float(), nullable=True),
        sa.Column(
            "ml_verdict",
            sa.Enum(
                "toxic",
                "not_toxic",
                "needs_ml_review",
                name="ml_verdict",
                native_enum=False,
            ),
            nullable=True,
        ),
        sa.Column("ml_model_version", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_comment_reports_comment_id", "comment_reports", ["comment_id"], unique=False)
    op.create_index("ix_comment_reports_reporter_id", "comment_reports", ["reporter_id"], unique=False)
    op.create_index("ix_comment_reports_status", "comment_reports", ["status"], unique=False)
    op.create_index("ix_comment_reports_moderation_verdict", "comment_reports", ["moderation_verdict"], unique=False)
    op.create_index("ix_comment_reports_reviewed_by_id", "comment_reports", ["reviewed_by_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_comment_reports_reviewed_by_id", table_name="comment_reports")
    op.drop_index("ix_comment_reports_moderation_verdict", table_name="comment_reports")
    op.drop_index("ix_comment_reports_status", table_name="comment_reports")
    op.drop_index("ix_comment_reports_reporter_id", table_name="comment_reports")
    op.drop_index("ix_comment_reports_comment_id", table_name="comment_reports")
    op.drop_table("comment_reports")

    op.drop_index("ix_comments_parent_comment_id", table_name="comments")
    op.drop_index("ix_comments_author_id", table_name="comments")
    op.drop_index("ix_comments_post_id", table_name="comments")
    op.drop_table("comments")

    op.drop_index("ix_posts_author_id", table_name="posts")
    op.drop_table("posts")

    op.drop_table("users")
