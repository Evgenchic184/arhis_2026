from __future__ import annotations

from datetime import datetime, timezone
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.deps import RequestUserContext, get_request_user_context, require_moderator
from src.app.core.database import get_db_session
from src.app.core.events import emit_domain_event
from src.app.core.queue import get_moderation_queue
from src.app.core.settings import get_settings
from src.app.core.monitoring import (
    increment_comments_hidden,
    increment_ml_routed,
    increment_reports_created,
    observe_moderation_decision_latency,
)
from src.app.models.comment import Comment
from src.app.models.enums import DecisionSource, ModerationVerdict, CommentVisibility, ReportStatus
from src.app.models.moderation import CommentReport
from src.app.schemas.moderation import (
    CommentReportCreate,
    CommentReportRead,
    ModerationDecisionCreate,
)
from src.app.services.user_counters import increment_user_counters
from src.app.services.moderation_routing import should_route_report_to_ml
from src.app.services.user_features import UserFeatureService
from src.feature_store.online import OnlineFeatureStore
from src.transformations.text import extract_required_text_features, preprocess_text

router = APIRouter(prefix="/moderation", tags=["moderation"])
logger = logging.getLogger(__name__)
user_feature_service = UserFeatureService()


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
        comment_body=report.comment.body,
        reason=report.reason,
        reason_text=report.reason_text,
        status=report.status,
        decision_source=report.decision_source,
        moderation_verdict=report.moderation_verdict,
        moderation_note=report.moderation_note,
        reviewed_by_id=report.reviewed_by_id,
        reviewed_by_name=_display_name(report.reviewed_by) if report.reviewed_by else None,
        reviewed_at=report.reviewed_at,
        ml_scored_at=report.ml_scored_at,
        ml_score=report.ml_score,
        ml_verdict=report.ml_verdict,
        ml_model_version=report.ml_model_version,
        ml_model_stage=report.ml_model_stage,
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

    settings = get_settings()
    feature_store = OnlineFeatureStore(redis_url=settings.redis_url, namespace=settings.feature_store_namespace)
    user_features = await feature_store.get_user_features(comment.author_id)
    report_id = uuid4()
    created_at = datetime.now(timezone.utc)
    route_to_ml = should_route_report_to_ml(report_id, settings.moderation_ml_route_rate)
    report = CommentReport(
        id=report_id,
        comment_id=comment_id,
        reporter_id=context.user_id,
        created_at=created_at,
        status=ReportStatus.queued_for_ml if route_to_ml else ReportStatus.pending,
        **payload.model_dump(),
    )
    db.add(report)
    await increment_user_counters(db, context.user_id, ["reports_count"])
    await emit_domain_event(
        db,
        event_type="comment_report_created",
        aggregate_type="comment_report",
        aggregate_id=str(report.id),
        payload={
            "comment_id": str(comment_id),
            "reporter_id": str(context.user_id),
            "reason": report.reason.value,
        },
        actor_id=str(context.user_id),
        actor_role=context.role.value,
    )
    if route_to_ml:
        await emit_domain_event(
            db,
            event_type="moderation_report_routed_to_ml",
            aggregate_type="comment_report",
            aggregate_id=str(report.id),
            payload={
                "report_id": str(report.id),
                "comment_id": str(comment_id),
                "comment_author_id": str(comment.author_id),
                "reporter_id": str(context.user_id),
                "reason": report.reason.value,
                "reason_text": report.reason_text,
                "comment_body": comment.body,
                "text_prepared": preprocess_text(comment.body),
                "text_features": extract_required_text_features(comment.body),
                "user_features": user_features,
                "created_at": created_at.isoformat(),
                "feature_config_version": settings.feature_config_version,
                "model_version": settings.ml_model_version,
            },
            actor_id=str(context.user_id),
            actor_role=context.role.value,
            topic=settings.kafka_moderation_ml_requests_topic,
        )
    else:
        await emit_domain_event(
            db,
            event_type="moderation_report_routed_to_manual",
            aggregate_type="comment_report",
            aggregate_id=str(report.id),
            payload={
                "report_id": str(report.id),
                "comment_id": str(comment_id),
                "comment_author_id": str(comment.author_id),
                "reporter_id": str(context.user_id),
                "reason": report.reason.value,
                "reason_text": report.reason_text,
                "comment_body": comment.body,
                "text_prepared": preprocess_text(comment.body),
                "text_features": extract_required_text_features(comment.body),
                "user_features": user_features,
                "created_at": created_at.isoformat(),
            },
            actor_id=str(context.user_id),
            actor_role=context.role.value,
        )
    await db.commit()
    increment_reports_created()
    increment_ml_routed("model" if route_to_ml else "manual")
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
    await user_feature_service.sync_user_features(
        db,
        comment.author_id,
        event_type="comment_report_created",
        metadata={
            "report_id": str(report.id),
            "reason": report.reason.value,
        },
    )
    if not route_to_ml:
        await get_moderation_queue().enqueue(
            {
                "report_id": str(report.id),
                "comment_id": str(comment_id),
                "reporter_id": str(context.user_id),
                "reason": report.reason.value,
                "created_at": created_at.isoformat(),
                "source": "user_report",
            }
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
    if report.status == ReportStatus.resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Report is already resolved.",
        )

    report.status = ReportStatus.resolved
    report.decision_source = DecisionSource.manual
    report.moderation_verdict = payload.verdict
    report.moderation_note = payload.note
    report.reviewed_by_id = context.user_id
    report.reviewed_at = datetime.now(timezone.utc)
    observe_moderation_decision_latency((report.reviewed_at - report.created_at).total_seconds())
    comment_hidden_now = False

    if payload.verdict == ModerationVerdict.toxic:
        if report.comment.visibility == CommentVisibility.visible:
            report.comment.visibility = CommentVisibility.hidden
            await increment_user_counters(db, report.comment.author_id, ["hidden_comments_count"])
            comment_hidden_now = True
            await emit_domain_event(
                db,
                event_type="comment_hidden",
                aggregate_type="comment",
                aggregate_id=str(report.comment.id),
                payload={
                    "post_id": str(report.comment.post_id),
                    "report_id": str(report.id),
                    "reason": report.reason.value,
                    "verdict": payload.verdict.value,
                },
                actor_id=str(context.user_id),
                actor_role=context.role.value,
            )

    await emit_domain_event(
        db,
        event_type="moderation_decision_created",
        aggregate_type="comment_report",
        aggregate_id=str(report.id),
        payload={
            "comment_id": str(report.comment_id),
            "verdict": payload.verdict.value,
            "note": payload.note,
            "reviewed_by_id": str(context.user_id),
        },
        actor_id=str(context.user_id),
        actor_role=context.role.value,
    )
    await db.commit()
    if comment_hidden_now:
        increment_comments_hidden()
    if payload.verdict == ModerationVerdict.toxic:
        await user_feature_service.sync_user_features(
            db,
            report.comment.author_id,
            event_type="comment_hidden",
            metadata={
                "report_id": str(report.id),
                "verdict": payload.verdict.value,
                "decision_source": report.decision_source.value if report.decision_source else "manual",
            },
        )

    if report.ml_verdict is not None and report.ml_verdict.value != payload.verdict.value:
        await user_feature_service.sync_user_features(
            db,
            report.comment.author_id,
            event_type="manual_overrule",
            metadata={
                "report_id": str(report.id),
                "ml_verdict": report.ml_verdict.value,
                "verdict": payload.verdict.value,
                "confidence": report.ml_score,
            },
        )
    else:
        await user_feature_service.sync_user_features(
            db,
            report.comment.author_id,
            event_type="manual_review",
            metadata={
                "report_id": str(report.id),
                "verdict": payload.verdict.value,
                "confidence": report.ml_score,
            },
        )
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
    await db.refresh(report)
    return _serialize_report(report)
