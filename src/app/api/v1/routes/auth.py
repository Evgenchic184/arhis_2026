from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.deps import RequestUserContext, get_request_user_context
from src.app.core.database import get_db_session
from src.app.core.events import emit_domain_event
from src.app.core.security import create_jwt, hash_password, verify_password
from src.app.core.settings import get_settings
from src.app.core.monitoring import refresh_users_total_once
from src.app.models.enums import UserRole
from src.app.models.user import User
from src.app.schemas.auth import TokenResponse, UserLogin, UserRegister
from src.app.schemas.users import UserRead
from src.app.services.user_features import UserFeatureService

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)
user_feature_service = UserFeatureService()


async def _username_exists(db: AsyncSession, username: str) -> bool:
    result = await db.execute(select(User.id).where(func.lower(User.username) == username.lower()).limit(1))
    return result.scalar_one_or_none() is not None


def _build_token(user: User) -> str:
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload = {
        "sub": str(user.id),
        "role": user.role.value,
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return create_jwt(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: UserRegister,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    username = payload.username.strip().lower()
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username is required.")
    if await _username_exists(db, username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists.")

    await db.execute(text("SELECT pg_advisory_xact_lock(20260513)"))
    result = await db.execute(
        select(func.count()).select_from(User).where(User.password_hash.is_not(None))
    )
    user_count = int(result.scalar_one() or 0)
    role = UserRole.admin if user_count == 0 else UserRole.user

    password_hash = hash_password(payload.password)
    salt = password_hash.split("$", 3)[2]
    user = User(
        username=username,
        email=None,
        role=role,
        is_active=True,
        password_hash=password_hash,
        password_salt=salt,
    )
    db.add(user)
    await emit_domain_event(
        db,
        event_type="user_registered",
        aggregate_type="user",
        aggregate_id=str(user.id),
        payload={"username": user.username, "role": user.role.value},
        actor_id=str(user.id),
        actor_role=user.role.value,
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists.") from exc
    await db.refresh(user)
    await refresh_users_total_once()
    await user_feature_service.sync_user_features(db, user.id, event_type="user_registered")

    token = _build_token(user)
    return TokenResponse(access_token=token, user=UserRead.model_validate(user))


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: UserLogin,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    username = payload.username.strip().lower()
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username is required.")
    result = await db.execute(select(User).where(func.lower(User.username) == username).limit(1))
    user = result.scalar_one_or_none()
    if user is None or not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is inactive.")

    user.last_login_at = datetime.now(timezone.utc)
    await emit_domain_event(
        db,
        event_type="user_logged_in",
        aggregate_type="user",
        aggregate_id=str(user.id),
        payload={"username": user.username, "role": user.role.value},
        actor_id=str(user.id),
        actor_role=user.role.value,
    )
    await db.commit()
    await db.refresh(user)
    await user_feature_service.sync_user_features(db, user.id, event_type="user_logged_in")

    token = _build_token(user)
    return TokenResponse(access_token=token, user=UserRead.model_validate(user))


@router.get("/me", response_model=UserRead)
async def read_me(
    context: RequestUserContext = Depends(get_request_user_context),
    db: AsyncSession = Depends(get_db_session),
) -> UserRead:
    user = await db.get(User, context.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return UserRead.model_validate(user)
