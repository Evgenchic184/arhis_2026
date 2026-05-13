from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from src.app.models.enums import UserRole


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    email: str | None
    role: UserRole
    is_active: bool
    display_name: str | None = None
    bio: str | None = None
    avatar_url: str | None = None
    last_login_at: datetime | None
    posts_count: int
    comments_count: int
    reports_count: int
    deleted_comments_count: int
    hidden_comments_count: int
    created_at: datetime
    updated_at: datetime


class UserRoleUpdate(BaseModel):
    role: UserRole
