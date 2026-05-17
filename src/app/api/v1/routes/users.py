from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.deps import RequestUserContext, require_admin
from src.app.core.database import get_db_session
from src.app.core.events import emit_domain_event
from src.app.models.user import User
from src.app.schemas.users import UserRead, UserRoleUpdate

router = APIRouter(prefix="/users", tags=["users"])
logger = logging.getLogger(__name__)


@router.get("", response_model=list[UserRead])
async def list_users(
    _: RequestUserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[UserRead]:
    result = await db.execute(select(User).order_by(User.created_at.asc()).limit(limit).offset(offset))
    return list(result.scalars().all())


@router.patch("/{user_id}/role", response_model=UserRead)
async def update_user_role(
    user_id: UUID,
    payload: UserRoleUpdate,
    context: RequestUserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
) -> UserRead:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    user.role = payload.role
    await emit_domain_event(
        db,
        event_type="user_role_updated",
        aggregate_type="user",
        aggregate_id=str(user.id),
        payload={"role": user.role.value},
        actor_id=str(context.user_id),
        actor_role=context.role.value,
    )
    await db.commit()
    await db.refresh(user)
    return user
