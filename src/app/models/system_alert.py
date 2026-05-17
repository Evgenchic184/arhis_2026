from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.app.models.base import Base, TimestampMixin, UUIDMixin
from src.app.models.enums import AlertStatus


class SystemAlert(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "system_alerts"

    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    status: Mapped[AlertStatus] = mapped_column(String(16), nullable=False, index=True)
    alertname: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    labels: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    annotations: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    receiver: Mapped[str | None] = mapped_column(String(128), nullable=True)
    generator_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    starts_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)
