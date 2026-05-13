from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.user import User

ALLOWED_COUNTERS = {
    "posts_count",
    "comments_count",
    "reports_count",
    "deleted_comments_count",
    "hidden_comments_count",
}


async def increment_user_counters(
    db: AsyncSession,
    user_id: UUID,
    counters: Iterable[str],
    amount: int = 1,
) -> None:
    payload: dict[str, object] = {}
    for counter in counters:
        if counter not in ALLOWED_COUNTERS:
            raise ValueError(f"Unsupported counter: {counter}")
        column = getattr(User, counter)
        payload[counter] = column + amount

    if not payload:
        return

    stmt = update(User).where(User.id == user_id).values(**payload)
    await db.execute(stmt)
