from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

try:
    import redis.asyncio as redis
except Exception:  # pragma: no cover - fallback when redis isn't installed
    redis = None

from src.app.core.settings import get_settings
from src.app.core.monitoring import set_queue_depth


class ModerationQueue:
    def __init__(self, redis_url: str, namespace: str = "arhis") -> None:
        self.namespace = namespace
        self._memory_queue: list[dict[str, Any]] = []
        self._client = redis.from_url(redis_url, decode_responses=True) if redis and redis_url else None

    @property
    def queue_key(self) -> str:
        return f"{self.namespace}:moderation:queue"

    async def enqueue(self, payload: dict[str, Any]) -> None:
        if self._client is None:
            self._memory_queue.append(payload)
            set_queue_depth(len(self._memory_queue))
            return
        try:
            await self._client.rpush(self.queue_key, json.dumps(payload, default=str))
            set_queue_depth(await self.size())
        except Exception:
            self._memory_queue.append(payload)
            set_queue_depth(len(self._memory_queue))

    async def dequeue(self) -> dict[str, Any] | None:
        if self._client is None:
            if not self._memory_queue:
                return None
            payload = self._memory_queue.pop(0)
            set_queue_depth(len(self._memory_queue))
            return payload

        try:
            item = await self._client.lpop(self.queue_key)
            set_queue_depth(await self.size())
            return json.loads(item) if item else None
        except Exception:
            if not self._memory_queue:
                return None
            payload = self._memory_queue.pop(0)
            set_queue_depth(len(self._memory_queue))
            return payload

    async def size(self) -> int:
        if self._client is None:
            return len(self._memory_queue)
        try:
            return int(await self._client.llen(self.queue_key))
        except Exception:
            return len(self._memory_queue)


@lru_cache(maxsize=1)
def get_moderation_queue() -> ModerationQueue:
    settings = get_settings()
    return ModerationQueue(settings.redis_url, namespace=settings.feature_store_namespace)
