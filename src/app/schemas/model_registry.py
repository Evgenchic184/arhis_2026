from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from src.app.models.enums import ModelVersionStatus


class ModelVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_name: str
    version: str
    status: ModelVersionStatus
    artifact_uri: str
    metadata_uri: str | None
    feature_config_version: int
    traffic_percent: int
    required_validation_accuracy: float
    validation_sample_size: int
    validation_accuracy: float | None
    validation_metrics: dict[str, Any] | None
    training_metadata: dict[str, Any] | None
    notes: str | None
    active_from_at: datetime | None
    promoted_at: datetime | None
    rolled_back_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ModelRegistryOverviewRead(BaseModel):
    model_name: str
    active_production: ModelVersionRead | None
    active_canary: ModelVersionRead | None
    versions: list[ModelVersionRead]

