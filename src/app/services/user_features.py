from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.core.settings import Settings, get_settings
from src.app.models.user import User
from src.feature_store.online import OnlineFeatureStore


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _window_buckets(now: datetime, event_at: datetime) -> tuple[bool, bool, bool]:
    age = now - event_at
    return age <= timedelta(days=1), age <= timedelta(days=7), age <= timedelta(days=30)


class UserFeatureService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._store = OnlineFeatureStore(redis_url=self.settings.redis_url, namespace=self.settings.feature_store_namespace)

    async def _load_user(self, db: AsyncSession, user_id: UUID) -> User | None:
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    def _base_snapshot(self, user: User, now: datetime) -> dict[str, Any]:
        account_age_days = max((now - user.created_at).days, 0)
        is_new_user = int(account_age_days < 7)
        comment_count_last_30d = int(user.comments_count or 0)
        reports_last_30d = int(user.reports_count or 0)
        deleted_comments_last_30d = int(user.deleted_comments_count or 0)
        hidden_comments_last_30d = int(user.hidden_comments_count or 0)

        reputation_score = _clamp(
            0.9
            - 0.06 * reports_last_30d
            - 0.04 * deleted_comments_last_30d
            - 0.05 * hidden_comments_last_30d
            + 0.0025 * comment_count_last_30d
            - (0.08 if is_new_user else 0.0),
        )

        return {
            "user_id": str(user.id),
            "is_new_user": is_new_user,
            "account_age_days": account_age_days,
            "reputation_score": reputation_score,
            "reports_last_24h": 0,
            "reports_last_7d": 0,
            "reports_last_30d": reports_last_30d,
            "deleted_comments_last_1d": 0,
            "deleted_comments_last_7d": 0,
            "deleted_comments_last_30d": deleted_comments_last_30d,
            "hidden_comments_last_1d": 0,
            "hidden_comments_last_7d": 0,
            "hidden_comments_last_30d": hidden_comments_last_30d,
            "comment_count_last_1d": 0,
            "comment_count_last_7d": 0,
            "comment_count_last_30d": comment_count_last_30d,
            "auto_action_count_last_30d": 0,
            "manual_overrule_count_last_30d": 0,
            "auto_action_rate_last_30d": 0.0,
            "manual_overrule_rate_last_30d": 0.0,
            "last_ml_confidence": 0.0,
            "last_ml_verdict": "unknown",
        }

    def _aggregate_events(self, events: list[dict[str, Any]], now: datetime) -> tuple[dict[str, int], dict[str, Any]]:
        buckets = defaultdict(int)
        latest_metadata: dict[str, Any] = {}
        latest_ts: datetime | None = None

        for event in events:
            event_type = str(event.get("event_type") or "")
            if not event_type:
                continue
            created_at = event.get("created_at")
            try:
                event_at = datetime.fromisoformat(str(created_at))
            except Exception:
                continue
            if event_at.tzinfo is None:
                event_at = event_at.replace(tzinfo=timezone.utc)
            within_1d, within_7d, within_30d = _window_buckets(now, event_at)
            if event_type == "comment_created":
                if within_1d:
                    buckets["comment_count_last_1d"] += 1
                if within_7d:
                    buckets["comment_count_last_7d"] += 1
                if within_30d:
                    buckets["comment_count_last_30d"] += 1
            elif event_type == "comment_deleted":
                if within_1d:
                    buckets["deleted_comments_last_1d"] += 1
                if within_7d:
                    buckets["deleted_comments_last_7d"] += 1
                if within_30d:
                    buckets["deleted_comments_last_30d"] += 1
            elif event_type == "comment_hidden":
                if within_1d:
                    buckets["hidden_comments_last_1d"] += 1
                if within_7d:
                    buckets["hidden_comments_last_7d"] += 1
                if within_30d:
                    buckets["hidden_comments_last_30d"] += 1
            elif event_type == "comment_report_created":
                if within_1d:
                    buckets["reports_last_24h"] += 1
                if within_7d:
                    buckets["reports_last_7d"] += 1
                if within_30d:
                    buckets["reports_last_30d"] += 1
            elif event_type == "ml_auto_action":
                if within_30d:
                    buckets["auto_action_count_last_30d"] += 1
            elif event_type == "manual_overrule":
                if within_30d:
                    buckets["manual_overrule_count_last_30d"] += 1

            if latest_ts is None or event_at >= latest_ts:
                latest_ts = event_at
                metadata = event.get("metadata") or {}
                if isinstance(metadata, Mapping):
                    latest_metadata = dict(metadata)

        return buckets, latest_metadata

    async def sync_user_features(
        self,
        db: AsyncSession,
        user_id: UUID,
        *,
        event_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        user = await self._load_user(db, user_id)
        if user is None:
            return {}

        now = datetime.now(timezone.utc)
        if event_type:
            await self._store.record_user_event(
                user_id,
                event_type=event_type,
                metadata=metadata,
                at=now,
            )

        events = await self._store.get_user_events(user_id)
        snapshot = self._base_snapshot(user, now)
        buckets, latest_metadata = self._aggregate_events(events, now)

        for key, value in buckets.items():
            snapshot[key] = int(value)

        reports_last_30d = max(int(snapshot.get("reports_last_30d", 0)), 0)
        comment_count_last_30d = max(int(snapshot.get("comment_count_last_30d", 0)), 0)
        deleted_comments_last_30d = max(int(snapshot.get("deleted_comments_last_30d", 0)), 0)
        hidden_comments_last_30d = max(int(snapshot.get("hidden_comments_last_30d", 0)), 0)
        auto_action_count_last_30d = max(int(snapshot.get("auto_action_count_last_30d", 0)), 0)
        manual_overrule_count_last_30d = max(int(snapshot.get("manual_overrule_count_last_30d", 0)), 0)

        if latest_metadata:
            if "confidence" in latest_metadata:
                confidence = _safe_float(latest_metadata.get("confidence"), default=float(snapshot.get("last_ml_confidence", 0.0)))
                snapshot["last_ml_confidence"] = confidence
            if "verdict" in latest_metadata:
                snapshot["last_ml_verdict"] = str(latest_metadata["verdict"])
            if "model_version" in latest_metadata:
                snapshot["last_model_version"] = str(latest_metadata["model_version"])

        if snapshot.get("last_ml_confidence") is None:
            snapshot["last_ml_confidence"] = float(snapshot.get("reputation_score", 0.0))

        if comment_count_last_30d > 0:
            snapshot["auto_action_rate_last_30d"] = auto_action_count_last_30d / comment_count_last_30d
            snapshot["manual_overrule_rate_last_30d"] = manual_overrule_count_last_30d / comment_count_last_30d
        else:
            snapshot["auto_action_rate_last_30d"] = 0.0
            snapshot["manual_overrule_rate_last_30d"] = 0.0

        snapshot["reputation_score"] = _clamp(
            float(snapshot.get("reputation_score", 0.0))
            - 0.04 * reports_last_30d
            - 0.03 * deleted_comments_last_30d
            - 0.04 * hidden_comments_last_30d
            - 0.02 * manual_overrule_count_last_30d
            + 0.01 * min(comment_count_last_30d, 100),
        )
        snapshot["reports_last_30d"] = reports_last_30d
        snapshot["last_updated_at"] = now.isoformat()

        await self._store.write_user_features(user_id, snapshot)
        return snapshot
