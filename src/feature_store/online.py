from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
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
        self._client = redis.from_url(redis_url, decode_responses=True) if redis and redis_url else None
        self._config_store = get_feature_config_store(redis_url=self.redis_url, namespace=self.namespace)

    def _key(self, user_id: str) -> str:
        return f"{self.namespace}:user:{user_id}:features"

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
