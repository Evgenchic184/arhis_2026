from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, ProgrammingError, SQLAlchemyError

from src.app.core.database import async_session_maker
from src.app.core.kafka import KafkaEventPublisher, KafkaEventRecord
from src.app.core.settings import get_settings
from src.app.models.event_outbox import DomainEventOutbox

logger = logging.getLogger(__name__)


async def relay_once(batch_size: int = 100) -> int:
    settings = get_settings()
    publisher = KafkaEventPublisher(settings.kafka_bootstrap_servers)
    if not publisher.enabled:
        logger.warning("kafka_not_configured", extra={"event": "kafka_not_configured"})
        return 0
    await publisher.start()

    processed = 0
    try:
        async with async_session_maker() as session:
            async with session.begin():
                stmt = (
                    select(DomainEventOutbox)
                    .where(DomainEventOutbox.published_at.is_(None))
                    .order_by(DomainEventOutbox.created_at.asc())
                    .limit(batch_size)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                events = list(result.scalars().all())

                for event in events:
                    try:
                        await publisher.publish(
                            KafkaEventRecord(
                                topic=event.topic or settings.kafka_domain_events_topic,
                                key=event.aggregate_id or str(event.id),
                                value={
                                    "event_id": str(event.id),
                                    "event_type": event.event_type,
                                    "aggregate_type": event.aggregate_type,
                                    "aggregate_id": event.aggregate_id,
                                    "actor_id": event.actor_id,
                                    "actor_role": event.actor_role,
                                    "request_id": event.request_id,
                                    "payload": event.payload,
                                    "created_at": event.created_at.isoformat(),
                                },
                            )
                        )
                        event.published_at = datetime.now(timezone.utc)
                        event.last_error = None
                        processed += 1
                    except Exception as exc:  # pragma: no cover - relay robustness
                        event.attempts += 1
                        event.last_error = str(exc)
                        logger.exception("event_outbox_publish_failed", extra={"event_id": str(event.id)})
    except (ProgrammingError, DBAPIError, SQLAlchemyError) as exc:
        logger.warning(
            "outbox_relay_waiting_for_schema",
            extra={"event": "outbox_relay_waiting_for_schema", "error": str(exc)},
        )
        return 0
    finally:
        await publisher.stop()

    return processed


async def main() -> None:
    while True:
        processed = await relay_once()
        if processed == 0:
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
