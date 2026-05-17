from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import redis.asyncio as redis
except Exception:  # pragma: no cover - fallback when redis isn't installed
    redis = None

from src.app.core.feature_config import get_feature_config_store
from src.feature_store.schemas import UserFeatureSnapshot


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return int(lowered == "true")
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


class OnlineFeatureStore:
    def __init__(self, redis_url: str | None = None, namespace: str = "arhis") -> None:
        self.redis_url = redis_url
        self.namespace = namespace
        self._memory_store: dict[str, dict[str, Any]] = {}
        self._memory_events: dict[str, list[dict[str, Any]]] = {}
        self._client = redis.from_url(redis_url, decode_responses=True) if redis and redis_url else None
        self._config_store = get_feature_config_store(redis_url=self.redis_url, namespace=self.namespace)

    def _key(self, user_id: str) -> str:
        return f"{self.namespace}:user:{user_id}:features"

    def _events_key(self, user_id: str) -> str:
        return f"{self.namespace}:user:{user_id}:events"

    async def get_user_features(self, user_id: str | int | None) -> dict[str, Any]:
        feature_config = await self._config_store.get()
        allowed_columns = set(feature_config.online_user_feature_columns)
        if user_id is None:
            snapshot = UserFeatureSnapshot(user_id="unknown", is_new_user=1).to_dict()
            return {key: value for key, value in snapshot.items() if key in allowed_columns or key == "user_id"}

        user_key = str(user_id)
        if self._client is None:
            snapshot = self._memory_store.get(user_key, UserFeatureSnapshot(user_id=user_key).to_dict())
            return {key: value for key, value in snapshot.items() if key in allowed_columns or key == "user_id"}

        try:
            payload = await self._client.hgetall(self._key(user_key))
            if not payload:
                snapshot = UserFeatureSnapshot(user_id=user_key).to_dict()
                return {key: value for key, value in snapshot.items() if key in allowed_columns or key == "user_id"}
            filtered = {
                key: _coerce_value(value)
                for key, value in payload.items()
                if key in allowed_columns or key == "user_id"
            }
            filtered.setdefault("user_id", user_key)
            return filtered
        except Exception:
            snapshot = self._memory_store.get(user_key, UserFeatureSnapshot(user_id=user_key).to_dict())
            return {key: value for key, value in snapshot.items() if key in allowed_columns or key == "user_id"}

    async def write_user_features(self, user_id: str | int, features: Mapping[str, Any]) -> None:
        user_key = str(user_id)
        payload = {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in features.items()}
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        if self._client is None:
            self._memory_store[user_key] = dict(payload)
            return
        try:
            await self._client.hset(self._key(user_key), mapping={key: str(value) for key, value in payload.items()})
        except Exception:
            self._memory_store[user_key] = dict(payload)

    async def record_user_event(
        self,
        user_id: str | int,
        *,
        event_type: str,
        metadata: Mapping[str, Any] | None = None,
        at: datetime | None = None,
        max_items: int = 500,
        prune_days: int = 30,
    ) -> None:
        user_key = str(user_id)
        event = {
            "event_type": event_type,
            "created_at": (at or datetime.now(timezone.utc)).isoformat(),
            "metadata": dict(metadata or {}),
        }

        if self._client is None:
            history = list(self._memory_events.get(user_key, []))
            history.append(event)
            cutoff = datetime.now(timezone.utc) - timedelta(days=prune_days)
            filtered = []
            for item in history[-max_items:]:
                created_at = item.get("created_at")
                try:
                    created_ts = datetime.fromisoformat(str(created_at))
                except Exception:
                    continue
                if created_ts.tzinfo is None:
                    created_ts = created_ts.replace(tzinfo=timezone.utc)
                if created_ts >= cutoff:
                    filtered.append(item)
            self._memory_events[user_key] = filtered
            return

        try:
            payload = await self._client.lrange(self._events_key(user_key), 0, max_items - 1)
            history = []
            for item in payload:
                try:
                    history.append(json.loads(item))
                except Exception:
                    continue
            history.append(event)
            cutoff = datetime.now(timezone.utc) - timedelta(days=prune_days)
            filtered = []
            for item in history[-max_items:]:
                created_at = item.get("created_at")
                try:
                    created_ts = datetime.fromisoformat(str(created_at))
                except Exception:
                    continue
                if created_ts.tzinfo is None:
                    created_ts = created_ts.replace(tzinfo=timezone.utc)
                if created_ts >= cutoff:
                    filtered.append(item)
            await self._client.delete(self._events_key(user_key))
            if filtered:
                await self._client.rpush(self._events_key(user_key), *[json.dumps(item, ensure_ascii=False) for item in filtered])
        except Exception:
            history = list(self._memory_events.get(user_key, []))
            history.append(event)
            self._memory_events[user_key] = history[-max_items:]

    async def get_user_events(self, user_id: str | int, max_items: int = 500) -> list[dict[str, Any]]:
        user_key = str(user_id)
        if self._client is None:
            return list(self._memory_events.get(user_key, []))
        try:
            payload = await self._client.lrange(self._events_key(user_key), 0, max_items - 1)
            events: list[dict[str, Any]] = []
            for item in payload:
                try:
                    events.append(json.loads(item))
                except Exception:
                    continue
            return events
        except Exception:
            return list(self._memory_events.get(user_key, []))

    async def record_moderation_event(
        self,
        user_id: str | int,
        *,
        action: str,
        confidence: float,
        model_version: str | None,
        is_manual: bool,
        reputation_delta: float = 0.0,
    ) -> dict[str, Any]:
        current = await self.get_user_features(user_id)
        current["last_ml_confidence"] = confidence
        current["last_ml_verdict"] = action
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        current["reputation_score"] = float(current.get("reputation_score", 0.0)) + reputation_delta

        if is_manual:
            current["manual_overrule_count_last_30d"] = int(current.get("manual_overrule_count_last_30d", 0)) + 1
        else:
            current["auto_action_count_last_30d"] = int(current.get("auto_action_count_last_30d", 0)) + 1

        total = max(int(current.get("comment_count_last_30d", 0)), 1)
        current["auto_action_rate_last_30d"] = int(current.get("auto_action_count_last_30d", 0)) / total
        current["manual_overrule_rate_last_30d"] = int(current.get("manual_overrule_count_last_30d", 0)) / total

        if model_version:
            current["last_model_version"] = model_version

        await self.write_user_features(user_id, current)
        return current
