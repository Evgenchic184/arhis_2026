from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from src.app.models.enums import AlertStatus


class AlertRead(BaseModel):
    id: UUID
    fingerprint: str
    status: AlertStatus
    alertname: str
    severity: str | None = None
    summary: str | None = None
    description: str | None = None
    labels: dict[str, Any]
    annotations: dict[str, Any]
    receiver: str | None = None
    generator_url: str | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    resolved_at: datetime | None = None
    received_at: datetime
    created_at: datetime
    updated_at: datetime
    is_active: bool
