from __future__ import annotations

from enum import Enum


class UserRole(str, Enum):
    user = "user"
    moderator = "moderator"
    admin = "admin"


class CommentVisibility(str, Enum):
    visible = "visible"
    hidden = "hidden"
    deleted = "deleted"


class ReportReason(str, Enum):
    harassment = "harassment"
    hate_speech = "hate_speech"
    spam = "spam"
    abuse = "abuse"
    other = "other"


class ReportStatus(str, Enum):
    pending = "pending"
    under_review = "under_review"
    resolved = "resolved"
    dismissed = "dismissed"


class ModerationVerdict(str, Enum):
    toxic = "toxic"
    not_toxic = "not_toxic"


class MLVerdict(str, Enum):
    toxic = "toxic"
    not_toxic = "not_toxic"
