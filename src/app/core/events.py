from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.app.core.request_context import get_request_id
from src.app.models.event_outbox import DomainEventOutbox

logger = logging.getLogger(__name__)


async def emit_domain_event(
    db: AsyncSession,
    *,
    event_type: str,
    aggregate_type: str,
    aggregate_id: str | None,
    payload: dict[str, Any],
    actor_id: str | None = None,
    actor_role: str | None = None,
    topic: str | None = None,
) -> DomainEventOutbox:
    event = DomainEventOutbox(
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        actor_id=actor_id,
        actor_role=actor_role,
        request_id=get_request_id(),
        topic=topic,
        payload=payload,
    )
    db.add(event)
    logger.info(
        event_type,
        extra={
            "event": event_type,
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "actor_id": actor_id,
            "actor_role": actor_role,
            "topic": topic,
        },
    )
    return event
