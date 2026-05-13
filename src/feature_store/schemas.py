from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class UserFeatureSnapshot:
    user_id: str
    is_new_user: int = 0
    account_age_days: int = 0
    reputation_score: float = 0.0
    reports_last_24h: int = 0
    reports_last_7d: int = 0
    reports_last_30d: int = 0
    deleted_comments_last_1d: int = 0
    deleted_comments_last_7d: int = 0
    deleted_comments_last_30d: int = 0
    hidden_comments_last_1d: int = 0
    hidden_comments_last_7d: int = 0
    hidden_comments_last_30d: int = 0
    comment_count_last_1d: int = 0
    comment_count_last_7d: int = 0
    comment_count_last_30d: int = 0
    auto_action_count_last_30d: int = 0
    manual_overrule_count_last_30d: int = 0
    auto_action_rate_last_30d: float = 0.0
    manual_overrule_rate_last_30d: float = 0.0
    last_ml_confidence: float = 0.0
    last_ml_verdict: str = "unknown"
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "is_new_user": self.is_new_user,
            "account_age_days": self.account_age_days,
            "reputation_score": self.reputation_score,
            "reports_last_24h": self.reports_last_24h,
            "reports_last_7d": self.reports_last_7d,
            "reports_last_30d": self.reports_last_30d,
            "deleted_comments_last_1d": self.deleted_comments_last_1d,
            "deleted_comments_last_7d": self.deleted_comments_last_7d,
            "deleted_comments_last_30d": self.deleted_comments_last_30d,
            "hidden_comments_last_1d": self.hidden_comments_last_1d,
            "hidden_comments_last_7d": self.hidden_comments_last_7d,
            "hidden_comments_last_30d": self.hidden_comments_last_30d,
            "comment_count_last_1d": self.comment_count_last_1d,
            "comment_count_last_7d": self.comment_count_last_7d,
            "comment_count_last_30d": self.comment_count_last_30d,
            "auto_action_count_last_30d": self.auto_action_count_last_30d,
            "manual_overrule_count_last_30d": self.manual_overrule_count_last_30d,
            "auto_action_rate_last_30d": self.auto_action_rate_last_30d,
            "manual_overrule_rate_last_30d": self.manual_overrule_rate_last_30d,
            "last_ml_confidence": self.last_ml_confidence,
            "last_ml_verdict": self.last_ml_verdict,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class CommentFeatureSnapshot:
    comment_id: str
    user_id: str
    text: str
    text_prepared: str
    label: int | None = None
    label_name: str | None = None
    label_source: str = "unknown"
    event_ts: datetime | None = None
    model_version: str | None = None
    confidence: float | None = None
    action: str | None = None
    user_features: dict[str, Any] = field(default_factory=dict)
    text_features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "comment_id": self.comment_id,
            "user_id": self.user_id,
            "text": self.text,
            "text_prepared": self.text_prepared,
            "label": self.label,
            "label_name": self.label_name,
            "label_source": self.label_source,
            "event_ts": self.event_ts,
            "model_version": self.model_version,
            "confidence": self.confidence,
            "action": self.action,
        }
        payload.update({f"user_{key}": value for key, value in self.user_features.items()})
        payload.update({f"text_{key}": value for key, value in self.text_features.items()})
        payload.update({f"meta_{key}": value for key, value in self.metadata.items()})
        return payload


@dataclass(slots=True)
class TrainingExample:
    snapshot: CommentFeatureSnapshot

    def to_flat_dict(self) -> dict[str, Any]:
        return self.snapshot.to_flat_dict()
