from __future__ import annotations

from datetime import datetime, timezone
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.deps import RequestUserContext, get_request_user_context, require_moderator
from src.app.core.database import get_db_session
from src.app.core.queue import get_moderation_queue
from src.app.models.comment import Comment
from src.app.models.enums import ModerationVerdict, CommentVisibility, ReportStatus
from src.app.models.moderation import CommentReport
from src.app.schemas.moderation import (
    CommentReportCreate,
    CommentReportRead,
    ModerationDecisionCreate,
)
from src.app.services.user_counters import increment_user_counters

router = APIRouter(prefix="/moderation", tags=["moderation"])
logger = logging.getLogger(__name__)


def _display_name(user) -> str:
    return user.display_name or user.username


def _serialize_report(report: CommentReport) -> CommentReportRead:
    return CommentReportRead(
        id=report.id,
        comment_id=report.comment_id,
        reporter_id=report.reporter_id,
        reporter_name=_display_name(report.reporter),
        comment_author_id=report.comment.author_id,
        comment_author_name=_display_name(report.comment.author),
        reason=report.reason,
        reason_text=report.reason_text,
        status=report.status,
        moderation_verdict=report.moderation_verdict,
        moderation_note=report.moderation_note,
        reviewed_by_id=report.reviewed_by_id,
        reviewed_by_name=_display_name(report.reviewed_by) if report.reviewed_by else None,
        reviewed_at=report.reviewed_at,
        ml_score=report.ml_score,
        ml_verdict=report.ml_verdict,
        ml_model_version=report.ml_model_version,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


@router.post(
    "/comments/{comment_id}/reports",
    response_model=CommentReportRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_comment_report(
    comment_id: UUID,
    payload: CommentReportCreate,
    db: AsyncSession = Depends(get_db_session),
    context: RequestUserContext = Depends(get_request_user_context),
) -> CommentReportRead:
    comment = await db.get(Comment, comment_id)
    if comment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found.")
    if comment.author_id == context.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot report your own comment.",
        )

    report = CommentReport(
        comment_id=comment_id,
        reporter_id=context.user_id,
        **payload.model_dump(),
    )
    db.add(report)
    await increment_user_counters(db, context.user_id, ["reports_count"])
    await db.commit()
    stmt = (
        select(CommentReport)
        .options(
            selectinload(CommentReport.reporter),
            selectinload(CommentReport.comment).selectinload(Comment.author),
        )
        .where(CommentReport.id == report.id)
    )
    result = await db.execute(stmt)
    report = result.scalar_one()
    await get_moderation_queue().enqueue(
        {
            "report_id": str(report.id),
            "comment_id": str(comment_id),
            "reporter_id": str(context.user_id),
            "reason": report.reason.value,
            "created_at": report.created_at.isoformat(),
            "source": "user_report",
        }
    )
    logger.info(
        "comment_report_created",
        extra={
            "event": "comment_report_created",
            "report_id": str(report.id),
            "comment_id": str(comment_id),
            "reporter_id": str(context.user_id),
            "reason": report.reason.value,
        },
    )
    return _serialize_report(report)


@router.get("/reports", response_model=list[CommentReportRead])
async def list_reports(
    db: AsyncSession = Depends(get_db_session),
    _: RequestUserContext = Depends(require_moderator),
    status_filter: ReportStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[CommentReportRead]:
    stmt = (
        select(CommentReport)
        .options(
            selectinload(CommentReport.reporter),
            selectinload(CommentReport.comment).selectinload(Comment.author),
            selectinload(CommentReport.reviewed_by),
        )
        .order_by(CommentReport.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if status_filter is not None:
        stmt = stmt.where(CommentReport.status == status_filter)
    result = await db.execute(stmt)
    return [_serialize_report(report) for report in result.scalars().all()]


@router.post("/reports/{report_id}/decision", response_model=CommentReportRead)
async def decide_report(
    report_id: UUID,
    payload: ModerationDecisionCreate,
    db: AsyncSession = Depends(get_db_session),
    context: RequestUserContext = Depends(require_moderator),
) -> CommentReportRead:
    stmt = (
        select(CommentReport)
        .options(
            selectinload(CommentReport.reporter),
            selectinload(CommentReport.comment).selectinload(Comment.post),
            selectinload(CommentReport.comment).selectinload(Comment.author),
            selectinload(CommentReport.reviewed_by),
        )
        .where(CommentReport.id == report_id)
    )
    result = await db.execute(stmt)
    report = result.scalar_one_or_none()
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    report.status = ReportStatus.resolved
    report.moderation_verdict = payload.verdict
    report.moderation_note = payload.note
    report.reviewed_by_id = context.user_id
    report.reviewed_at = datetime.now(timezone.utc)

    if payload.verdict == ModerationVerdict.toxic:
        if report.comment.visibility == CommentVisibility.visible:
            report.comment.visibility = CommentVisibility.hidden
            await increment_user_counters(db, report.comment.author_id, ["hidden_comments_count"])

    await db.commit()
    stmt = (
        select(CommentReport)
        .options(
            selectinload(CommentReport.reporter),
            selectinload(CommentReport.comment).selectinload(Comment.post),
            selectinload(CommentReport.comment).selectinload(Comment.author),
            selectinload(CommentReport.reviewed_by),
        )
        .where(CommentReport.id == report.id)
    )
    result = await db.execute(stmt)
    report = result.scalar_one()
    logger.info(
        "moderation_decision_created",
        extra={
            "event": "moderation_decision_created",
            "report_id": str(report_id),
            "comment_id": str(report.comment_id),
            "moderator_id": str(context.user_id),
            "verdict": payload.verdict.value,
        },
    )
    return _serialize_report(report)
