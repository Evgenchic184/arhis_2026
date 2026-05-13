from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Enum as SAEnum

from src.app.models.base import Base, TimestampMixin, UUIDMixin
from src.app.models.enums import UserRole

if TYPE_CHECKING:
    from src.app.models.comment import Comment
    from src.app.models.moderation import CommentReport
    from src.app.models.post import Post


class User(UUIDMixin, TimestampMixin, Base):
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    password_salt: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, name="user_role", native_enum=False),
        default=UserRole.user,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    bio: Mapped[str | None] = mapped_column(String(500), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    posts_count: Mapped[int] = mapped_column(default=0, nullable=False)
    comments_count: Mapped[int] = mapped_column(default=0, nullable=False)
    reports_count: Mapped[int] = mapped_column(default=0, nullable=False)
    deleted_comments_count: Mapped[int] = mapped_column(default=0, nullable=False)
    hidden_comments_count: Mapped[int] = mapped_column(default=0, nullable=False)

    posts: Mapped[list["Post"]] = relationship(back_populates="author")
    comments: Mapped[list["Comment"]] = relationship(back_populates="author")
    reports_created: Mapped[list["CommentReport"]] = relationship(
        back_populates="reporter",
        foreign_keys="CommentReport.reporter_id",
    )
    reports_reviewed: Mapped[list["CommentReport"]] = relationship(
        back_populates="reviewed_by",
        foreign_keys="CommentReport.reviewed_by_id",
    )
