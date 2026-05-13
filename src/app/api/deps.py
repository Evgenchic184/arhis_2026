from __future__ import annotations

from uuid import UUID

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from src.app.core.database import get_db_session
from src.app.core.security import InvalidTokenError, decode_jwt
from src.app.core.settings import get_settings
from src.app.models.enums import UserRole
from src.app.models.user import User


class RequestUserContext(BaseModel):
    user_id: UUID
    username: str
    role: UserRole = UserRole.user
    email: str | None = None
    is_active: bool = True


bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth", description="JWT Bearer authentication")


def _extract_bearer_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization credentials are required.",
        )
    if credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization must use Bearer token.",
        )
    return credentials.credentials


async def get_request_user_context(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> RequestUserContext:
    settings = get_settings()
    token = _extract_bearer_token(credentials)
    try:
        payload = decode_jwt(token, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    user_id_raw = payload.get("sub")
    if not user_id_raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token subject is missing.")

    try:
        user = await db.get(User, UUID(str(user_id_raw)))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token subject is invalid.") from exc

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is inactive.")

    return RequestUserContext(
        user_id=user.id,
        username=user.username,
        role=user.role,
        email=user.email,
        is_active=user.is_active,
    )


def require_moderator(
    context: RequestUserContext = Depends(get_request_user_context),
) -> RequestUserContext:
    if context.role not in {UserRole.moderator, UserRole.admin}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Moderator access is required.",
        )
    return context


def require_admin(
    context: RequestUserContext = Depends(get_request_user_context),
) -> RequestUserContext:
    if context.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is required.",
        )
    return context
