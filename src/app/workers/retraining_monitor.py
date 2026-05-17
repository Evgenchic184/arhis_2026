from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

from src.app.core.database import async_session_maker
from src.app.core.metrics_server import start_metrics_server
from src.app.core.monitoring import (
    set_data_freshness_seconds,
    set_data_new_token_share,
    set_data_psi,
    set_ml_stage_avg_confidence,
    clear_ml_stage_metrics,
    set_ml_stage_manual_accuracy,
    set_ml_stage_manual_review_rate,
    set_ml_stage_manual_reviews_total,
    set_ml_stage_reports_total,
    set_model_sample_manual_accuracy,
    set_model_sample_manual_count,
    set_retraining_trigger,
)
from src.app.core.settings import get_settings
from src.app.models.comment import Comment
from src.app.models.enums import CommentVisibility, DecisionSource
from src.app.models.moderation import CommentReport
from src.app.services.drift import compute_new_token_share, compute_psi_from_bins
from src.app.services.event_archive import EventArchiveQuery, EventArchiveStore
from src.app.services.model_registry import ModelRegistryService

logger = logging.getLogger(__name__)


class RetrainingMonitor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model_registry = ModelRegistryService(self.settings)
        self._archive = EventArchiveStore(self.settings)
        self._last_retraining_started_at: float = 0.0
        self._retraining_running = False
        self._active_stage_metric_keys: set[tuple[str, str]] = set()
        self._stage_metrics_lock = asyncio.Lock()

    async def _get_active_models(self) -> tuple[Any | None, Any | None]:
        async with async_session_maker() as db:
            try:
                versions = await self._model_registry.list_active_versions(db, self.settings.model_registry_model_name)
            except (ProgrammingError, SQLAlchemyError):
                logger.info(
                    "retraining_monitor_waiting_for_registry",
                    extra={"event": "retraining_monitor_waiting_for_registry"},
                )
                return None, None
            production = next((item for item in versions if item.status.value == "production"), None)
            canary = next((item for item in versions if item.status.value == "canary"), None)
            return production, canary

    def _extract_numeric_series(self, frame, feature: str):
        if feature in frame.columns:
            return pd.to_numeric(frame[feature], errors="coerce")
        payload_col = f"payload_{feature}"
        if payload_col in frame.columns:
            return pd.to_numeric(frame[payload_col], errors="coerce")
        nested_payload_columns = (
            "payload_text_features",
            "payload_user_features",
            "payload_features",
            "text_features",
            "user_features",
        )
        for nested_column in nested_payload_columns:
            if nested_column not in frame.columns:
                continue

            def _extract_nested(value: Any) -> Any:
                if isinstance(value, dict):
                    return value.get(feature)
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                    except Exception:
                        return None
                    if isinstance(parsed, dict):
                        return parsed.get(feature)
                return None

            nested_series = frame[nested_column].map(_extract_nested)
            if nested_series.notna().any():
                return pd.to_numeric(nested_series, errors="coerce")
        return pd.Series(dtype=float)

    def _extract_texts(self, frame) -> list[str]:
        texts: list[str] = []
        for _, row in frame.iterrows():
            text = row.get("payload_comment_body") or row.get("payload_text_prepared") or ""
            if text:
                texts.append(str(text))
        return texts

    async def _filter_deleted_comments(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "payload_comment_id" not in frame.columns:
            return frame

        comment_ids = [
            str(comment_id)
            for comment_id in frame["payload_comment_id"].dropna().astype(str).tolist()
            if str(comment_id)
        ]
        if not comment_ids:
            return frame

        async with async_session_maker() as db:
            stmt = (
                select(Comment.id)
                .where(Comment.id.in_(comment_ids))
                .where(Comment.visibility != CommentVisibility.deleted)
            )
            result = await db.execute(stmt)
            active_comment_ids = {str(comment_id) for comment_id in result.scalars().all()}

        filtered = frame[frame["payload_comment_id"].astype(str).isin(active_comment_ids)].copy()
        deleted_rows = int(len(frame) - len(filtered))
        if deleted_rows:
            logger.info(
                "retraining_monitor_deleted_comments_filtered",
                extra={
                    "event": "retraining_monitor_deleted_comments_filtered",
                    "deleted_rows": deleted_rows,
                },
            )
        return filtered

    async def _manual_sample_accuracy(self) -> tuple[float, int]:
        window_start = datetime.now(timezone.utc) - timedelta(hours=self.settings.retraining_manual_sample_window_hours)
        async with async_session_maker() as db:
            stmt = (
                select(CommentReport)
                .where(CommentReport.reviewed_at.is_not(None))
                .where(CommentReport.reviewed_at >= window_start)
                .where(CommentReport.decision_source == DecisionSource.manual)
                .where(CommentReport.ml_verdict.is_not(None))
                .where(CommentReport.moderation_verdict.is_not(None))
            )
            result = await db.execute(stmt)
            reports = result.scalars().all()
        if not reports:
            return 0.0, 0
        matches = sum(int(report.ml_verdict.value == report.moderation_verdict.value) for report in reports)
        return matches / len(reports), len(reports)

    async def _collect_stage_metrics(
        self,
        since: datetime,
        model_versions: list[str],
    ) -> dict[tuple[str, str], dict[str, float]]:
        if not model_versions:
            return {}

        async with async_session_maker() as db:
            stmt = (
                select(
                    CommentReport.ml_model_version,
                    CommentReport.ml_model_stage,
                    CommentReport.decision_source,
                    CommentReport.ml_verdict,
                    CommentReport.moderation_verdict,
                    CommentReport.ml_score,
                )
                .where(CommentReport.ml_model_version.in_(model_versions))
                .where(CommentReport.ml_scored_at.is_not(None))
                .where(CommentReport.ml_scored_at >= since)
                .order_by(CommentReport.ml_scored_at.asc())
            )
            result = await db.execute(stmt)
            rows = result.all()

        stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {
                "scored_count": 0.0,
                "manual_review_count": 0.0,
                "manual_match_count": 0.0,
                "confidence_sum": 0.0,
            }
        )
        for model_version, stage, decision_source, ml_verdict, moderation_verdict, ml_score in rows:
            version = str(model_version or "")
            stage_name = str(stage or "")
            if not version or not stage_name:
                continue

            entry = stats[(version, stage_name)]
            entry["scored_count"] += 1.0
            if ml_score is not None:
                entry["confidence_sum"] += float(ml_score)

            decision_source_value = getattr(decision_source, "value", decision_source)
            if (
                decision_source_value == DecisionSource.manual.value
                and ml_verdict is not None
                and moderation_verdict is not None
            ):
                entry["manual_review_count"] += 1.0
                ml_value = getattr(ml_verdict, "value", ml_verdict)
                moderation_value = getattr(moderation_verdict, "value", moderation_verdict)
                if ml_value == moderation_value:
                    entry["manual_match_count"] += 1.0

        for key, entry in stats.items():
            scored = entry["scored_count"]
            manual = entry["manual_review_count"]
            entry["manual_review_rate"] = manual / scored if scored else 0.0
            entry["manual_accuracy"] = entry["manual_match_count"] / manual if manual else 0.0
            entry["avg_confidence"] = entry["confidence_sum"] / scored if scored else 0.0
        return stats

    async def _update_stage_metrics(
        self,
        production: Any | None,
        canary: Any | None,
    ) -> dict[tuple[str, str], dict[str, float]]:
        active_versions = [snapshot.version for snapshot in (production, canary) if snapshot is not None]
        if not active_versions:
            return {}

        now = datetime.now(timezone.utc)
        if canary and canary.active_from_at is not None:
            since = canary.active_from_at
        else:
            since = now - timedelta(minutes=5)

        stats = await self._collect_stage_metrics(since, active_versions)
        current_keys = set(stats.keys())
        for stale_version, stale_stage in self._active_stage_metric_keys - current_keys:
            clear_ml_stage_metrics(stale_version, stale_stage)
        self._active_stage_metric_keys = current_keys
        for (model_version, stage), entry in stats.items():
            set_ml_stage_reports_total(model_version, stage, int(entry["scored_count"]))
            set_ml_stage_manual_reviews_total(model_version, stage, int(entry["manual_review_count"]))
            set_ml_stage_manual_review_rate(model_version, stage, float(entry["manual_review_rate"]))
            set_ml_stage_manual_accuracy(model_version, stage, float(entry["manual_accuracy"]))
            set_ml_stage_avg_confidence(model_version, stage, float(entry["avg_confidence"]))
        return stats

    def _publish_monitor_metrics(self, metrics: dict[str, Any]) -> None:
        psi_values = metrics.get("psi_values", {})
        data_freshness_seconds = float(metrics.get("data_freshness_seconds", 0.0))
        new_token_share = float(metrics.get("new_token_share", 0.0))
        manual_accuracy = float(metrics.get("manual_sample_accuracy", 0.0))
        manual_count = int(metrics.get("manual_sample_count", 0))

        set_data_freshness_seconds(data_freshness_seconds)
        set_data_new_token_share(new_token_share)
        for feature, value in psi_values.items():
            set_data_psi(feature, value)
        set_model_sample_manual_accuracy(manual_accuracy)
        set_model_sample_manual_count(manual_count)

    async def _refresh_stage_metrics(
        self,
        production: Any | None,
        canary: Any | None,
    ) -> dict[tuple[str, str], dict[str, float]]:
        async with self._stage_metrics_lock:
            return await self._update_stage_metrics(production, canary)

    async def _maybe_auto_manage_canary(
        self,
        production: Any | None,
        canary: Any | None,
        stats: dict[tuple[str, str], dict[str, float]],
    ) -> bool:
        if not self.settings.model_registry_auto_promote_enabled:
            return False
        if production is None or canary is None:
            return False
        if canary.active_from_at is None:
            return False

        age_seconds = (datetime.now(timezone.utc) - canary.active_from_at).total_seconds()
        if age_seconds < self.settings.model_registry_auto_promote_min_age_seconds:
            return False

        canary_entry = stats.get((canary.version, "canary"))
        production_entry = stats.get((production.version, "production"))
        if canary_entry is None or production_entry is None:
            return False

        if canary_entry["scored_count"] < float(self.settings.model_registry_auto_promote_min_reports):
            return False
        if (
            canary_entry["manual_review_count"] < float(self.settings.model_registry_auto_promote_min_manual_samples)
            or production_entry["manual_review_count"] < float(self.settings.model_registry_auto_promote_min_manual_samples)
        ):
            return False

        canary_better = (
            canary_entry["manual_accuracy"] >= production_entry["manual_accuracy"]
            and canary_entry["manual_review_rate"] <= production_entry["manual_review_rate"]
        )
        canary_worse = (
            canary_entry["manual_accuracy"] < production_entry["manual_accuracy"]
            and canary_entry["manual_review_rate"] > production_entry["manual_review_rate"]
        )

        async with async_session_maker() as db:
            if canary_better:
                promoted = await self._model_registry.promote_version(
                    db,
                    model_name=self.settings.model_registry_model_name,
                    version=canary.version,
                )
                await db.commit()
                logger.warning(
                    "canary_auto_promoted",
                    extra={
                        "event": "canary_auto_promoted",
                        "model_version": promoted.version,
                        "production_version": production.version,
                        "canary_metrics": canary_entry,
                        "production_metrics": production_entry,
                    },
                )
                return True

            if canary_worse:
                await self._model_registry.mark_rolled_back(db, canary.id)
                await db.commit()
                logger.warning(
                    "canary_auto_rolled_back",
                    extra={
                        "event": "canary_auto_rolled_back",
                        "model_version": canary.version,
                        "production_version": production.version,
                        "canary_metrics": canary_entry,
                        "production_metrics": production_entry,
                    },
                )
                return True
        return False

    async def _retraining_cycle(self) -> None:
        if not self.settings.retraining_enabled:
            return

        metrics = await self._compute_metrics()
        if not metrics.get("ready"):
            logger.info("retraining_monitor_not_ready", extra={"event": "retraining_monitor_not_ready"})
            return

        psi_values = metrics.get("psi_values", {})
        max_psi = max(psi_values.values(), default=0.0)
        new_token_share = float(metrics.get("new_token_share", 0.0))
        manual_accuracy = float(metrics.get("manual_sample_accuracy", 0.0))
        manual_count = int(metrics.get("manual_sample_count", 0))
        production = metrics.get("production")
        canary = metrics.get("canary")

        self._publish_monitor_metrics(metrics)

        await self._refresh_stage_metrics(production, canary)

        triggers = {
            "psi": max_psi >= self.settings.retraining_psi_threshold,
            "new_token_share": new_token_share >= self.settings.retraining_new_token_share_threshold,
            "manual_accuracy": manual_count >= self.settings.retraining_manual_accuracy_min_samples
            and manual_accuracy < self.settings.retraining_manual_accuracy_threshold
            and (1.0 - manual_accuracy) >= self.settings.retraining_manual_accuracy_drop_threshold,
        }
        for reason, active in triggers.items():
            set_retraining_trigger(reason, active)

        if not any(triggers.values()):
            return

        now = time.monotonic()
        if self._retraining_running or (now - self._last_retraining_started_at) < self.settings.retraining_cooldown_seconds:
            logger.info(
                "retraining_trigger_detected",
                extra={
                    "event": "retraining_trigger_detected",
                    "psi": max_psi,
                    "new_token_share": new_token_share,
                    "manual_accuracy": manual_accuracy,
                    "manual_count": manual_count,
                    "cooldown_active": self._retraining_running,
                },
            )
            return

        logger.warning(
            "retraining_triggered",
            extra={
                "event": "retraining_triggered",
                "psi": max_psi,
                "new_token_share": new_token_share,
                "manual_accuracy": manual_accuracy,
                "manual_count": manual_count,
                "psi_values": psi_values,
            },
        )
        await self._launch_retraining()

    async def _canary_management_cycle(self) -> None:
        if not self.settings.retraining_enabled or not self.settings.model_registry_auto_promote_enabled:
            return

        metrics = await self._compute_metrics()
        if not metrics.get("ready"):
            return

        production = metrics.get("production")
        canary = metrics.get("canary")

        self._publish_monitor_metrics(metrics)

        stage_stats = await self._refresh_stage_metrics(production, canary)
        await self._maybe_auto_manage_canary(production, canary, stage_stats)

    async def _compute_metrics(self) -> dict[str, Any]:
        production, canary = await self._get_active_models()
        if production is None:
            return {"ready": False}

        try:
            model = await self._model_registry.load_model(production.artifact_uri)
        except Exception as exc:
            logger.info(
                "retraining_monitor_waiting_for_model_artifact",
                extra={"event": "retraining_monitor_waiting_for_model_artifact", "error": str(exc)},
            )
            return {"ready": False}

        training_metadata = production.training_metadata or {}
        feature_profiles = training_metadata.get("feature_profiles", {})
        baseline_vocab = set()
        try:
            vectorizer = model.named_steps["features"].named_transformers_["text"]
            vocabulary = getattr(vectorizer, "vocabulary_", {})
            baseline_vocab = set(vocabulary.keys()) if isinstance(vocabulary, dict) else set(vocabulary)
        except Exception:
            baseline_vocab = set()

        query = EventArchiveQuery(
            bucket_name=self.settings.s3_bucket_name,
            prefix=self.settings.parquet_sink_prefix,
            event_types=[
                "moderation_report_routed_to_manual",
                "moderation_report_routed_to_ml",
                "comment_deleted",
            ],
            days_back=max(1, self.settings.retraining_data_window_hours // 24 or 1),
        )
        try:
            recent_events = await self._archive.read_events(query)
        except Exception as exc:
            logger.info(
                "retraining_monitor_waiting_for_archive",
                extra={"event": "retraining_monitor_waiting_for_archive", "error": str(exc)},
            )
            return {"ready": False}
        if recent_events.empty:
            return {"ready": False}

        recent_events = await self._filter_deleted_comments(recent_events)
        if recent_events.empty:
            return {"ready": False}

        metrics: dict[str, Any] = {
            "ready": True,
            "production_version": production.version,
            "artifact_uri": production.artifact_uri,
            "production": production,
            "canary": canary,
        }

        created_at = pd.to_datetime(recent_events["created_at"], utc=True, errors="coerce")
        if not created_at.empty:
            freshness_seconds = (pd.Timestamp.now(tz="UTC") - created_at.max()).total_seconds()
            metrics["data_freshness_seconds"] = float(max(freshness_seconds, 0.0))
        else:
            metrics["data_freshness_seconds"] = 0.0

        texts = self._extract_texts(recent_events)
        metrics["new_token_share"] = compute_new_token_share(texts, baseline_vocab)

        psi_values: dict[str, float] = {}
        for feature, profile in feature_profiles.items():
            reference = profile.get("bin_distribution")
            bin_edges = profile.get("bin_edges")
            if not reference or not bin_edges:
                continue
            current_series = self._extract_numeric_series(recent_events, feature)
            if current_series.empty:
                continue
            psi_values[feature] = compute_psi_from_bins(
                current_series.tolist(),
                bin_edges=bin_edges,
                reference_distribution=reference,
            )
        metrics["psi_values"] = psi_values

        manual_accuracy, manual_count = await self._manual_sample_accuracy()
        metrics["manual_sample_accuracy"] = manual_accuracy
        metrics["manual_sample_count"] = manual_count
        return metrics

    async def _launch_retraining(self) -> None:
        self._retraining_running = True
        self._last_retraining_started_at = time.monotonic()
        logger.warning("retraining_started", extra={"event": "retraining_started"})
        process = await asyncio.create_subprocess_exec("python", "-m", "src.pipeline.retrain_model")
        await process.wait()
        self._retraining_running = False
        logger.warning(
            "retraining_finished",
            extra={"event": "retraining_finished", "returncode": process.returncode},
        )

    async def tick(self) -> None:
        await self._retraining_cycle()

    async def run(self) -> None:
        async def _run_retraining_loop() -> None:
            while True:
                try:
                    await self._retraining_cycle()
                except Exception:
                    logger.exception("retraining_monitor_failed")
                await asyncio.sleep(self.settings.retraining_monitor_interval_seconds)

        async def _run_canary_management_loop() -> None:
            while True:
                try:
                    await self._canary_management_cycle()
                except Exception:
                    logger.exception("retraining_monitor_canary_management_failed")
                await asyncio.sleep(self.settings.model_registry_auto_promote_check_interval_seconds)

        await asyncio.gather(_run_retraining_loop(), _run_canary_management_loop())


async def main() -> None:
    worker = RetrainingMonitor()
    start_metrics_server(worker.settings.retraining_monitor_metrics_port)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
