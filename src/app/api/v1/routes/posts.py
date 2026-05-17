from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.deps import RequestUserContext, get_request_user_context
from src.app.core.database import get_db_session
from src.app.core.events import emit_domain_event
from src.app.core.monitoring import increment_posts_created
from src.app.models.post import Post
from src.app.schemas.posts import PostCreate, PostRead, PostUpdate
from src.app.services.user_counters import increment_user_counters
from src.app.services.user_features import UserFeatureService

router = APIRouter(prefix="/posts", tags=["posts"])
logger = logging.getLogger(__name__)
user_feature_service = UserFeatureService()


def _serialize_post(post: Post) -> PostRead:
    author_name = post.author.display_name or post.author.username
    return PostRead(
        id=post.id,
        title=post.title,
        body=post.body,
        author_id=post.author_id,
        author_name=author_name,
        is_published=post.is_published,
        comments_count=post.comments_count,
        created_at=post.created_at,
        updated_at=post.updated_at,
    )


@router.post("", response_model=PostRead, status_code=status.HTTP_201_CREATED)
async def create_post(
    payload: PostCreate,
    db: AsyncSession = Depends(get_db_session),
    context: RequestUserContext = Depends(get_request_user_context),
) -> PostRead:
    post = Post(author_id=context.user_id, **payload.model_dump())
    db.add(post)
    await increment_user_counters(db, context.user_id, ["posts_count"])
    await emit_domain_event(
        db,
        event_type="post_created",
        aggregate_type="post",
        aggregate_id=str(post.id),
        payload={"title": post.title, "is_published": post.is_published},
        actor_id=str(context.user_id),
        actor_role=context.role.value,
    )
    await db.commit()
    increment_posts_created()
    await user_feature_service.sync_user_features(db, context.user_id, event_type="post_created")
    stmt = select(Post).options(selectinload(Post.author)).where(Post.id == post.id)
    result = await db.execute(stmt)
    post = result.scalar_one()
    return _serialize_post(post)


@router.get("", response_model=list[PostRead])
async def list_posts(
    db: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[PostRead]:
    result = await db.execute(
        select(Post)
        .options(selectinload(Post.author))
        .order_by(Post.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return [_serialize_post(post) for post in result.scalars().all()]


@router.get("/{post_id}", response_model=PostRead)
async def get_post(
    post_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> PostRead:
    stmt = select(Post).options(selectinload(Post.author)).where(Post.id == post_id)
    result = await db.execute(stmt)
    post = result.scalar_one_or_none()
    if post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found.")
    return _serialize_post(post)


@router.patch("/{post_id}", response_model=PostRead)
async def update_post(
    post_id: UUID,
    payload: PostUpdate,
    db: AsyncSession = Depends(get_db_session),
    context: RequestUserContext = Depends(get_request_user_context),
) -> PostRead:
    post = await db.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found.")
    if post.author_id != context.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not post owner.")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(post, field, value)

    await db.commit()
    stmt = select(Post).options(selectinload(Post.author)).where(Post.id == post.id)
    result = await db.execute(stmt)
    return _serialize_post(result.scalar_one())


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    context: RequestUserContext = Depends(get_request_user_context),
) -> None:
    post = await db.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found.")
    if post.author_id != context.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not post owner.")

    await db.delete(post)
    await emit_domain_event(
        db,
        event_type="post_deleted",
        aggregate_type="post",
        aggregate_id=str(post_id),
        payload={"author_id": str(post.author_id)},
        actor_id=str(context.user_id),
        actor_role=context.role.value,
    )
    await db.commit()
