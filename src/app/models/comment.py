from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.base import Base, TimestampMixin, UUIDMixin
from src.app.models.enums import CommentVisibility
from sqlalchemy import Enum as SAEnum

if TYPE_CHECKING:
    from src.app.models.moderation import CommentReport
    from src.app.models.post import Post
    from src.app.models.user import User


class Comment(UUIDMixin, TimestampMixin, Base):
    post_id: Mapped[UUID] = mapped_column(
        ForeignKey("posts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    author_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_comment_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("comments.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)
    visibility: Mapped[CommentVisibility] = mapped_column(
        SAEnum(CommentVisibility, name="comment_visibility", native_enum=False),
        default=CommentVisibility.visible,
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    post: Mapped["Post"] = relationship(back_populates="comments")
    author: Mapped["User"] = relationship(back_populates="comments")
    parent_comment: Mapped["Comment | None"] = relationship(
        "Comment",
        remote_side="Comment.id",
        back_populates="replies",
    )
    replies: Mapped[list["Comment"]] = relationship(back_populates="parent_comment")
    reports: Mapped[list["CommentReport"]] = relationship(
        back_populates="comment",
        cascade="all, delete-orphan",
    )
