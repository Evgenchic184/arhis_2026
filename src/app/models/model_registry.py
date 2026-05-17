from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Enum as SAEnum

from src.app.models.base import Base, TimestampMixin, UUIDMixin
from src.app.models.enums import ModelVersionStatus


class ModelVersion(UUIDMixin, TimestampMixin, Base):
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_versions_model_name_version"),
    )

    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[ModelVersionStatus] = mapped_column(
        SAEnum(ModelVersionStatus, name="model_version_status", native_enum=False),
        nullable=False,
        index=True,
    )
    artifact_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    metadata_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    feature_config_version: Mapped[int] = mapped_column(Integer, nullable=False)
    traffic_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    required_validation_accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    validation_sample_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    training_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    active_from_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

