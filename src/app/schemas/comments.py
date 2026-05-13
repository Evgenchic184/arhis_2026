from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.app.models.enums import CommentVisibility


class CommentBase(BaseModel):
    body: str = Field(min_length=1)
    parent_comment_id: UUID | None = None


class CommentCreate(CommentBase):
    pass


class CommentUpdate(BaseModel):
    body: str = Field(min_length=1)


class CommentRead(CommentBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    post_id: UUID
    author_id: UUID
    author_name: str
    visibility: CommentVisibility
    is_deleted: bool
    deleted_at: datetime | None
    created_at: datetime
    updated_at: datetime
