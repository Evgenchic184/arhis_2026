from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, ProgrammingError, SQLAlchemyError
from sqlalchemy.orm import selectinload

try:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.structs import TopicPartition
except Exception:  # pragma: no cover - optional dependency in dev environments
    AIOKafkaConsumer = None
    TopicPartition = None

from src.app.core.database import async_session_maker
from src.app.core.events import emit_domain_event
from src.app.core.metrics_server import start_metrics_server
from src.app.core.monitoring import (
    increment_ml_auto_action,
    increment_ml_inference,
    increment_ml_manual_escalation,
    increment_ml_model_stage,
    observe_moderation_decision_latency,
    observe_ml_confidence,
    observe_ml_inference_latency,
)
from src.app.core.queue import get_moderation_queue
from src.app.core.settings import get_settings
from src.app.models.comment import Comment
from src.app.models.enums import CommentVisibility, DecisionSource, MLVerdict, ModerationVerdict, ReportStatus
from src.app.models.moderation import CommentReport
from src.app.services.model_registry import ModelRegistryService, ModelSnapshot
from src.app.services.moderation_routing import should_sample_manual_review
from src.app.services.user_features import UserFeatureService
from src.feature_store.config import load_runtime_feature_config
from src.feature_store.online import OnlineFeatureStore
from src.transformations.text import extract_required_text_features, preprocess_text

logger = logging.getLogger(__name__)


class ModerationModelWorker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._legacy_model = None
        self._model_path = Path(self.settings.ml_model_path)
        self._feature_store = OnlineFeatureStore(
            redis_url=self.settings.redis_url,
            namespace=self.settings.feature_store_namespace,
        )
        self._user_feature_service = UserFeatureService(self.settings)
        self._model_registry = ModelRegistryService(self.settings)
        self._registry_cache: list[ModelSnapshot] = []
        self._registry_cache_loaded_at: float = 0.0
        self._registry_cache_ttl_seconds = 30.0
        self._model_cache: dict[str, Any] = {}

    async def _load_model(self) -> None:
        while True:
            if self._model_path.exists():
                self._legacy_model = await asyncio.to_thread(joblib.load, self._model_path)
                logger.info(
                    "ml_model_loaded",
                    extra={
                        "event": "ml_model_loaded",
                        "model_path": str(self._model_path),
                        "model_version": self.settings.ml_model_version,
                    },
                )
                return
            logger.warning(
                "ml_model_not_found",
                extra={"event": "ml_model_not_found", "model_path": str(self._model_path)},
            )
            await asyncio.sleep(5)

    async def _get_registry_models(self) -> list[ModelSnapshot]:
        if not self.settings.model_registry_enabled:
            return []
        now = time.monotonic()
        if self._registry_cache and now - self._registry_cache_loaded_at < self._registry_cache_ttl_seconds:
            return list(self._registry_cache)

        try:
            async with async_session_maker() as db:
                snapshots = await self._model_registry.list_active_versions(db, self.settings.model_registry_model_name)
        except (ProgrammingError, DBAPIError, SQLAlchemyError):
            logger.info(
                "ml_registry_waiting_for_schema",
                extra={"event": "ml_registry_waiting_for_schema"},
            )
            return []
        self._registry_cache = list(snapshots)
        self._registry_cache_loaded_at = now
        return list(self._registry_cache)

    async def _load_registry_model(self, snapshot: ModelSnapshot):
        if snapshot.artifact_uri in self._model_cache:
            return self._model_cache[snapshot.artifact_uri]
        model = await self._model_registry.load_model(snapshot.artifact_uri)
        self._model_cache[snapshot.artifact_uri] = model
        logger.info(
            "ml_registry_model_loaded",
            extra={
                "event": "ml_registry_model_loaded",
                "model_name": snapshot.model_name,
                "model_version": snapshot.version,
                "status": snapshot.status.value,
                "artifact_uri": snapshot.artifact_uri,
            },
        )
        return model

    async def _build_inference_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        feature_config = await load_runtime_feature_config(
            redis_url=self.settings.redis_url,
            namespace=self.settings.feature_store_namespace,
        )
        text = str(payload.get("comment_body") or "")
        prepared_text = str(payload.get("text_prepared") or preprocess_text(text))
        text_features = payload.get("text_features") or extract_required_text_features(text)
        user_id = payload.get("comment_author_id")
        user_features = await self._feature_store.get_user_features(user_id)

        row: dict[str, Any] = {feature_config.text_column: prepared_text, "user_id": user_id}
        for column in feature_config.inference_feature_columns:
            if column == feature_config.text_column:
                continue
            if column in text_features:
                row[column] = text_features[column]
            elif column in user_features:
                row[column] = user_features[column]
            else:
                row[column] = 0

        for column in feature_config.inference_feature_columns:
            row.setdefault(column, 0)
        row.setdefault(feature_config.text_column, prepared_text)
        return pd.DataFrame([row])

    async def _predict(self, model, payload: dict[str, Any]) -> tuple[str, float]:
        if model is None:
            await self._load_model()
        if model is None:
            model = self._legacy_model
        assert model is not None

        frame = await self._build_inference_frame(payload)
        predicted = await asyncio.to_thread(model.predict, frame)
        if hasattr(model, "predict_proba"):
            probabilities = await asyncio.to_thread(model.predict_proba, frame)
            confidence = float(max(probabilities[0]))
        else:
            confidence = 1.0
        verdict = "toxic" if int(predicted[0]) == 1 else "not_toxic"
        return verdict, confidence

    async def _build_report_payload_from_db(self, report: CommentReport) -> dict[str, Any]:
        text = report.comment.body
        user_features = await self._feature_store.get_user_features(report.comment.author_id)
        return {
            "report_id": str(report.id),
            "comment_id": str(report.comment_id),
            "comment_author_id": str(report.comment.author_id),
            "comment_body": text,
            "reason": report.reason.value,
            "reason_text": report.reason_text,
            "text_prepared": preprocess_text(text),
            "text_features": extract_required_text_features(text),
            "user_features": user_features,
        }

    async def _sweep_pending_reports(self, limit: int = 20) -> int:
        async with async_session_maker() as db:
            stmt = (
                select(CommentReport)
                .options(
                    selectinload(CommentReport.comment).selectinload(Comment.author),
                    selectinload(CommentReport.reporter),
                )
                .where(CommentReport.status == ReportStatus.queued_for_ml)
                .where(CommentReport.ml_scored_at.is_(None))
                .order_by(CommentReport.created_at.asc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            reports = result.scalars().all()

        processed = 0
        for report in reports:
            try:
                await self.handle_report(await self._build_report_payload_from_db(report))
                processed += 1
            except Exception:
                logger.exception("ml_inference_replay_failed", extra={"report_id": str(report.id)})
        return processed

    async def _resolve_model_for_report(self, report_uuid: UUID) -> tuple[str, str, Any, int]:
        registry_models = await self._get_registry_models()
        production = next((snapshot for snapshot in registry_models if snapshot.status.value == "production"), None)
        canary = next((snapshot for snapshot in registry_models if snapshot.status.value == "canary"), None)

        if canary is not None and (
            production is None or self._model_registry.route_stage_for_report(report_uuid, canary.traffic_percent) == "canary"
        ):
            return "canary", canary.version, await self._load_registry_model(canary), canary.traffic_percent

        if production is not None:
            return "production", production.version, await self._load_registry_model(production), 100

        if self._legacy_model is None:
            await self._load_model()
        assert self._legacy_model is not None
        return "legacy", self.settings.ml_model_version, self._legacy_model, 100

    async def handle_report(self, payload: dict[str, Any]) -> None:
        event_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
        report_id = event_payload.get("report_id")
        comment_id = event_payload.get("comment_id")
        if not report_id or not comment_id:
            logger.warning("ml_inference_invalid_payload", extra={"payload": payload})
            return

        report_uuid = UUID(str(report_id))

        stage, model_version, model, traffic_percent = await self._resolve_model_for_report(report_uuid)
        increment_ml_model_stage(stage, model_version=model_version)

        start_time = time.perf_counter()
        verdict, confidence = await self._predict(model, event_payload)
        latency = time.perf_counter() - start_time
        observe_ml_confidence(confidence, model_version=model_version, stage=stage)
        observe_ml_inference_latency(latency, model_version=model_version, stage=stage)
        increment_ml_inference(verdict, model_version=model_version, stage=stage)

        low_threshold = self.settings.moderation_ml_confidence_threshold_low
        high_threshold = self.settings.moderation_ml_confidence_threshold_high
        route_manual = confidence < high_threshold
        manual_reason = "uncertain_band" if confidence >= low_threshold else "low_confidence"
        if confidence >= high_threshold:
            route_manual = should_sample_manual_review(
                report_uuid,
                self.settings.moderation_ml_manual_sample_rate,
            )
            manual_reason = "sampled_review"

        async with async_session_maker() as db:
            stmt = (
                select(CommentReport)
                .options(
                    selectinload(CommentReport.comment),
                    selectinload(CommentReport.comment).selectinload(Comment.author),
                    selectinload(CommentReport.reporter),
                )
                .where(CommentReport.id == report_id)
            )
            result = await db.execute(stmt)
            report = result.scalar_one_or_none()
            if report is None:
                logger.warning("ml_report_not_found", extra={"report_id": report_id})
                return
            if report.ml_scored_at is not None:
                logger.info(
                    "ml_report_already_scored",
                    extra={"report_id": str(report.id), "ml_scored_at": report.ml_scored_at.isoformat()},
                )
                return

            report.ml_score = confidence
            report.ml_verdict = MLVerdict.toxic if verdict == "toxic" else MLVerdict.not_toxic
            report.ml_model_version = model_version
            report.ml_model_stage = stage
            report.ml_scored_at = datetime.now(timezone.utc)

            if route_manual:
                report.status = ReportStatus.under_review
                report.decision_source = None
                await emit_domain_event(
                    db,
                    event_type="moderation_report_escalated_to_manual",
                    aggregate_type="comment_report",
                    aggregate_id=str(report.id),
                    payload={
                        "report_id": str(report.id),
                        "comment_id": str(report.comment_id),
                        "confidence": confidence,
                        "verdict": verdict,
                        "reason": manual_reason,
                        "model_version": model_version,
                        "model_stage": stage,
                        "traffic_percent": traffic_percent,
                    },
                    actor_id=None,
                    actor_role="system",
                )
                await get_moderation_queue().enqueue(
                    {
                        "report_id": str(report.id),
                        "comment_id": str(report.comment_id),
                        "reason": report.reason.value,
                        "source": "ml_escalation",
                        "confidence": confidence,
                        "verdict": verdict,
                        "model_version": model_version,
                        "model_stage": stage,
                        "traffic_percent": traffic_percent,
                    }
                )
                increment_ml_manual_escalation(manual_reason, model_version=model_version, stage=stage)
                await db.commit()
                await self._user_feature_service.sync_user_features(
                    db,
                    report.comment.author_id,
                    event_type="manual_review",
                    metadata={
                        "report_id": str(report.id),
                        "ml_verdict": verdict,
                        "confidence": confidence,
                        "model_version": model_version,
                        "model_stage": stage,
                    },
                )
                return

            report.status = ReportStatus.resolved
            report.decision_source = DecisionSource.ml_auto
            report.reviewed_at = datetime.now(timezone.utc)
            report.reviewed_by_id = None
            report.moderation_verdict = ModerationVerdict.toxic if verdict == "toxic" else ModerationVerdict.not_toxic
            report.moderation_note = f"Auto decision by ML model {model_version} ({stage})."
            observe_moderation_decision_latency((report.reviewed_at - report.created_at).total_seconds())

            if verdict == "toxic":
                if report.comment.visibility == CommentVisibility.visible:
                    report.comment.visibility = CommentVisibility.hidden
                    await emit_domain_event(
                        db,
                        event_type="comment_hidden",
                        aggregate_type="comment",
                        aggregate_id=str(report.comment.id),
                        payload={
                            "post_id": str(report.comment.post_id),
                            "report_id": str(report.id),
                            "reason": report.reason.value,
                            "verdict": verdict,
                            "decision_source": "ml_auto",
                            "model_stage": stage,
                            "model_version": model_version,
                        },
                        actor_id=None,
                        actor_role="system",
                    )
                increment_ml_auto_action("hide", model_version=model_version, stage=stage)
            else:
                increment_ml_auto_action("allow", model_version=model_version, stage=stage)

            await emit_domain_event(
                db,
                event_type="moderation_decision_created",
                aggregate_type="comment_report",
                aggregate_id=str(report.id),
                payload={
                    "comment_id": str(report.comment_id),
                    "verdict": verdict,
                    "note": report.moderation_note,
                    "reviewed_by_id": None,
                    "decision_source": "ml_auto",
                    "confidence": confidence,
                    "model_version": model_version,
                    "model_stage": stage,
                },
                actor_id=None,
                actor_role="system",
            )
            await db.commit()
            if verdict == "toxic":
                await self._user_feature_service.sync_user_features(
                    db,
                    report.comment.author_id,
                    event_type="comment_hidden",
                    metadata={
                        "report_id": str(report.id),
                        "ml_verdict": verdict,
                        "confidence": confidence,
                        "model_version": model_version,
                        "model_stage": stage,
                    },
                )
            await self._user_feature_service.sync_user_features(
                db,
                report.comment.author_id,
                event_type="ml_auto_action",
                metadata={
                    "report_id": str(report.id),
                    "ml_verdict": verdict,
                    "confidence": confidence,
                    "model_version": model_version,
                    "model_stage": stage,
                },
            )

    async def _replay_loop(self) -> None:
        while True:
            try:
                await self._sweep_pending_reports()
            except Exception:
                logger.exception("ml_inference_replay_loop_failed")
            await asyncio.sleep(30)

    async def run(self) -> None:
        if not self.settings.kafka_bootstrap_servers or AIOKafkaConsumer is None or TopicPartition is None:
            raise RuntimeError("Kafka consumer is not configured.")

        consumer = AIOKafkaConsumer(
            self.settings.kafka_moderation_ml_requests_topic,
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            group_id="arhis-ml-inference",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            key_deserializer=lambda value: value.decode("utf-8") if value else None,
        )
        replay_task: asyncio.Task[None] | None = None

        try:
            while True:
                try:
                    await consumer.start()
                    replay_task = asyncio.create_task(self._replay_loop())
                    break
                except Exception as exc:
                    logger.info(
                        "ml_inference_waiting_for_dependencies",
                        extra={"event": "ml_inference_waiting_for_dependencies", "error": str(exc)},
                    )
                    await asyncio.sleep(5)

            while True:
                try:
                    message = await consumer.getone()
                except Exception as exc:
                    logger.warning(
                        "ml_inference_consumer_waiting",
                        extra={"event": "ml_inference_consumer_waiting", "error": str(exc)},
                    )
                    await asyncio.sleep(5)
                    continue
                try:
                    await self.handle_report(message.value)
                    await consumer.commit({TopicPartition(message.topic, message.partition): message.offset + 1})
                except Exception:
                    logger.exception("ml_inference_failed", extra={"payload": message.value})
        finally:
            if replay_task is not None:
                replay_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await replay_task
            await consumer.stop()


async def main() -> None:
    worker = ModerationModelWorker()
    start_metrics_server(worker.settings.ml_worker_metrics_port)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
