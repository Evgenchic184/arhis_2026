from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    from aiokafka import AIOKafkaProducer
except Exception:  # pragma: no cover - optional dependency in dev environments
    AIOKafkaProducer = None


@dataclass(slots=True)
class KafkaEventRecord:
    topic: str
    key: str
    value: dict[str, Any]


class KafkaEventPublisher:
    def __init__(self, bootstrap_servers: str) -> None:
        self.bootstrap_servers = bootstrap_servers
        self._producer = None

    @property
    def enabled(self) -> bool:
        return bool(self.bootstrap_servers and AIOKafkaProducer is not None)

    async def start(self) -> None:
        if not self.enabled or self._producer is not None:
            return
        self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def publish(self, record: KafkaEventRecord) -> None:
        if not self.enabled:
            return
        if self._producer is None:
            await self.start()
        assert self._producer is not None
        await self._producer.send_and_wait(
            record.topic,
            value=json.dumps(record.value, ensure_ascii=False, default=str).encode("utf-8"),
            key=record.key.encode("utf-8"),
        )
