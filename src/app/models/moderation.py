from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Enum as SAEnum

from src.app.models.base import Base, TimestampMixin, UUIDMixin
from src.app.models.enums import MLVerdict, ModerationVerdict, ReportReason, ReportStatus

if TYPE_CHECKING:
    from src.app.models.comment import Comment
    from src.app.models.user import User


class CommentReport(UUIDMixin, TimestampMixin, Base):
    comment_id: Mapped[UUID] = mapped_column(
        ForeignKey("comments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reporter_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reason: Mapped[ReportReason] = mapped_column(
        SAEnum(ReportReason, name="report_reason", native_enum=False),
        nullable=False,
    )
    reason_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ReportStatus] = mapped_column(
        SAEnum(ReportStatus, name="report_status", native_enum=False),
        default=ReportStatus.pending,
        nullable=False,
        index=True,
    )
    moderation_verdict: Mapped[ModerationVerdict | None] = mapped_column(
        SAEnum(ModerationVerdict, name="moderation_verdict", native_enum=False),
        nullable=True,
        index=True,
    )
    moderation_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ml_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_verdict: Mapped[MLVerdict | None] = mapped_column(
        SAEnum(MLVerdict, name="ml_verdict", native_enum=False),
        nullable=True,
    )
    ml_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    comment: Mapped["Comment"] = relationship(back_populates="reports")
    reporter: Mapped["User"] = relationship(
        back_populates="reports_created",
        foreign_keys=[reporter_id],
    )
    reviewed_by: Mapped["User | None"] = relationship(
        back_populates="reports_reviewed",
        foreign_keys=[reviewed_by_id],
    )
