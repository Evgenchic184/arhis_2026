from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.app.models.enums import (
    DecisionSource,
    MLVerdict,
    ModerationVerdict,
    ReportReason,
    ReportStatus,
)


class CommentReportCreate(BaseModel):
    reason: ReportReason
    reason_text: str | None = Field(default=None, max_length=4000)


class CommentReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    comment_id: UUID
    reporter_id: UUID
    reporter_name: str
    comment_author_id: UUID
    comment_author_name: str
    comment_body: str
    reason: ReportReason
    reason_text: str | None
    status: ReportStatus
    decision_source: DecisionSource | None
    moderation_verdict: ModerationVerdict | None
    moderation_note: str | None
    reviewed_by_id: UUID | None
    reviewed_by_name: str | None
    reviewed_at: datetime | None
    ml_scored_at: datetime | None
    ml_score: float | None
    ml_verdict: MLVerdict | None
    ml_model_version: str | None
    ml_model_stage: str | None
    created_at: datetime
    updated_at: datetime


class ModerationDecisionCreate(BaseModel):
    verdict: ModerationVerdict
    note: str | None = Field(default=None, max_length=4000)
