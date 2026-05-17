from __future__ import annotations

from datetime import datetime, timezone
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.deps import RequestUserContext, get_request_user_context
from src.app.core.database import get_db_session
from src.app.core.events import emit_domain_event
from src.app.core.monitoring import increment_comments_created
from src.app.models.comment import Comment
from src.app.models.enums import CommentVisibility, UserRole
from src.app.models.post import Post
from src.app.schemas.comments import CommentCreate, CommentRead, CommentUpdate
from src.app.services.user_counters import increment_user_counters
from src.app.services.user_features import UserFeatureService

router = APIRouter(tags=["comments"])
logger = logging.getLogger(__name__)
user_feature_service = UserFeatureService()

DELETED_COMMENT_BODY = "Комментарий удален"
HIDDEN_COMMENT_BODY = "Комментарий скрыт модератором"


def _serialize_comment(comment: Comment) -> CommentRead:
    author_name = comment.author.display_name or comment.author.username
    if comment.visibility == CommentVisibility.deleted:
        body = DELETED_COMMENT_BODY
    elif comment.visibility == CommentVisibility.hidden:
        body = HIDDEN_COMMENT_BODY
    else:
        body = comment.body

    return CommentRead(
        id=comment.id,
        body=body,
        parent_comment_id=comment.parent_comment_id,
        post_id=comment.post_id,
        author_id=comment.author_id,
        author_name=author_name,
        visibility=comment.visibility,
        is_deleted=comment.is_deleted,
        deleted_at=comment.deleted_at,
        created_at=comment.created_at,
        updated_at=comment.updated_at,
    )


@router.post(
    "/posts/{post_id}/comments",
    response_model=CommentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_comment(
    post_id: UUID,
    payload: CommentCreate,
    db: AsyncSession = Depends(get_db_session),
    context: RequestUserContext = Depends(get_request_user_context),
) -> CommentRead:
    post = await db.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found.")

    if payload.parent_comment_id is not None:
        parent = await db.get(Comment, payload.parent_comment_id)
        if parent is None or parent.post_id != post_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent comment is invalid for this post.",
            )

    comment = Comment(
        post_id=post_id,
        author_id=context.user_id,
        **payload.model_dump(),
    )
    db.add(comment)
    await increment_user_counters(db, context.user_id, ["comments_count"])
    post.comments_count += 1
    await emit_domain_event(
        db,
        event_type="comment_created",
        aggregate_type="comment",
        aggregate_id=str(comment.id),
        payload={
            "post_id": str(post_id),
            "parent_comment_id": str(payload.parent_comment_id) if payload.parent_comment_id else None,
            "visibility": comment.visibility.value,
        },
        actor_id=str(context.user_id),
        actor_role=context.role.value,
    )
    await db.commit()
    increment_comments_created()
    await user_feature_service.sync_user_features(db, context.user_id, event_type="comment_created")
    stmt = select(Comment).options(selectinload(Comment.author)).where(Comment.id == comment.id)
    result = await db.execute(stmt)
    comment = result.scalar_one()
    return _serialize_comment(comment)


@router.get("/posts/{post_id}/comments", response_model=list[CommentRead])
async def list_post_comments(
    post_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> list[CommentRead]:
    post = await db.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found.")

    result = await db.execute(
        select(Comment)
        .options(selectinload(Comment.author))
        .where(Comment.post_id == post_id)
        .order_by(Comment.created_at.desc())
    )
    return [_serialize_comment(comment) for comment in result.scalars().all()]


@router.patch("/comments/{comment_id}", response_model=CommentRead)
async def update_comment(
    comment_id: UUID,
    payload: CommentUpdate,
    db: AsyncSession = Depends(get_db_session),
    context: RequestUserContext = Depends(get_request_user_context),
) -> CommentRead:
    comment = await db.get(Comment, comment_id)
    if comment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found.")
    if comment.author_id != context.user_id and context.role not in {UserRole.moderator, UserRole.admin}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed.")
    if comment.visibility == CommentVisibility.deleted:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Comment is deleted.")

    comment.body = payload.body
    await emit_domain_event(
        db,
        event_type="comment_updated",
        aggregate_type="comment",
        aggregate_id=str(comment.id),
        payload={"post_id": str(comment.post_id)},
        actor_id=str(context.user_id),
        actor_role=context.role.value,
    )
    await db.commit()
    stmt = select(Comment).options(selectinload(Comment.author)).where(Comment.id == comment.id)
    result = await db.execute(stmt)
    comment = result.scalar_one()
    return _serialize_comment(comment)


@router.delete("/comments/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comment(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    context: RequestUserContext = Depends(get_request_user_context),
) -> None:
    comment = await db.get(Comment, comment_id)
    if comment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found.")
    if comment.author_id != context.user_id and context.role not in {UserRole.moderator, UserRole.admin}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed.")
    if comment.visibility == CommentVisibility.deleted:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Comment is already deleted.")

    comment.visibility = CommentVisibility.deleted
    comment.is_deleted = True
    comment.deleted_at = datetime.now(timezone.utc)
    await increment_user_counters(db, comment.author_id, ["deleted_comments_count"])
    await emit_domain_event(
        db,
        event_type="comment_deleted",
        aggregate_type="comment",
        aggregate_id=str(comment.id),
        payload={"post_id": str(comment.post_id), "deleted_at": comment.deleted_at.isoformat()},
        actor_id=str(context.user_id),
        actor_role=context.role.value,
    )
    await db.commit()
    await user_feature_service.sync_user_features(db, comment.author_id, event_type="comment_deleted")
