from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from sqlalchemy import func, select

from src.app.models.user import User

logger = logging.getLogger(__name__)

REQUESTS_TOTAL = Counter(
    "arhis_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "arhis_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)

MODERATION_QUEUE_DEPTH = Gauge(
    "arhis_moderation_queue_depth",
    "Current moderation queue depth.",
)

ML_REPORTS_ROUTED_TOTAL = Counter(
    "arhis_ml_reports_routed_total",
    "Total moderation reports routed to ML or manual queues.",
    ["route"],
)

ML_INFERENCE_TOTAL = Counter(
    "arhis_ml_inference_total",
    "Total number of moderation reports processed by ML.",
    ["verdict"],
)

ML_INFERENCE_BY_MODEL_TOTAL = Counter(
    "arhis_ml_inference_by_model_total",
    "Total number of moderation reports processed by ML grouped by model version and stage.",
    ["model_version", "stage", "verdict"],
)

ML_AUTO_ACTION_TOTAL = Counter(
    "arhis_ml_auto_action_total",
    "Total number of ML auto actions applied to moderation reports.",
    ["action"],
)

ML_AUTO_ACTION_BY_MODEL_TOTAL = Counter(
    "arhis_ml_auto_action_by_model_total",
    "Total number of ML auto actions applied grouped by model version and stage.",
    ["model_version", "stage", "action"],
)

ML_MANUAL_ESCALATION_TOTAL = Counter(
    "arhis_ml_manual_escalation_total",
    "Total number of ML escalations to manual review.",
    ["reason"],
)

ML_MANUAL_ESCALATION_BY_MODEL_TOTAL = Counter(
    "arhis_ml_manual_escalation_by_model_total",
    "Total number of ML escalations to manual review grouped by model version and stage.",
    ["model_version", "stage", "reason"],
)

ML_MODEL_STAGE_TOTAL = Counter(
    "arhis_ml_model_stage_total",
    "Total moderation reports scored by serving stage.",
    ["stage"],
)

ML_MODEL_STAGE_BY_MODEL_TOTAL = Counter(
    "arhis_ml_model_stage_by_model_total",
    "Total moderation reports scored by serving stage and model version.",
    ["model_version", "stage"],
)

MODEL_REGISTRY_VALIDATION_TOTAL = Counter(
    "arhis_model_registry_validation_total",
    "Total model registry validation results.",
    ["result"],
)

MODEL_REGISTRY_DEPLOYMENT_TOTAL = Counter(
    "arhis_model_registry_deployment_total",
    "Total model registry deployment transitions.",
    ["status"],
)

DATA_PSI = Gauge(
    "arhis_data_psi",
    "Population stability index for monitored numeric features.",
    ["feature"],
)

DATA_NEW_TOKEN_SHARE = Gauge(
    "arhis_data_new_token_share",
    "Share of tokens not present in the baseline vocabulary.",
)

DATA_FRESHNESS_SECONDS = Gauge(
    "arhis_data_freshness_seconds",
    "Age of the newest parquet event in seconds.",
)

MODEL_SAMPLE_MANUAL_ACCURACY = Gauge(
    "arhis_model_sample_manual_accuracy",
    "Accuracy on sampled manual review rows.",
)

MODEL_SAMPLE_MANUAL_COUNT = Gauge(
    "arhis_model_sample_manual_count",
    "Number of sampled manual review rows used in monitoring.",
)

RETRAINING_TRIGGER_ACTIVE = Gauge(
    "arhis_retraining_trigger_active",
    "Whether the retraining trigger is active.",
    ["reason"],
)

ML_INFERENCE_LATENCY = Histogram(
    "arhis_ml_inference_latency_seconds",
    "ML inference latency in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

ML_CONFIDENCE = Histogram(
    "arhis_ml_confidence",
    "Confidence values returned by the ML model.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0),
)

ML_CONFIDENCE_BY_MODEL = Histogram(
    "arhis_ml_confidence_by_model",
    "Confidence values returned by the ML model grouped by model version and stage.",
    ["model_version", "stage"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0),
)

MODEL_CONFIDENCE = Histogram(
    "arhis_model_confidence",
    "Confidence values returned by the ML model.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0),
)

USERS_TOTAL = Gauge(
    "arhis_users_total",
    "Current number of users in the system.",
)

POSTS_CREATED_TOTAL = Counter(
    "arhis_posts_created_total",
    "Total number of created posts.",
)

COMMENTS_CREATED_TOTAL = Counter(
    "arhis_comments_created_total",
    "Total number of created comments.",
)

REPORTS_CREATED_TOTAL = Counter(
    "arhis_reports_created_total",
    "Total number of created moderation reports.",
)

COMMENTS_HIDDEN_TOTAL = Counter(
    "arhis_comments_hidden_total",
    "Total number of comments hidden by moderation.",
)

MODERATION_DECISION_LATENCY_SECONDS = Histogram(
    "arhis_moderation_decision_latency_seconds",
    "Time between report creation and moderation verdict in seconds.",
    buckets=(1, 5, 10, 30, 60, 300, 900, 1800, 3600, 7200, 14400, 28800),
)

ML_INFERENCE_LATENCY_BY_MODEL = Histogram(
    "arhis_ml_inference_latency_by_model_seconds",
    "ML inference latency in seconds grouped by model version and stage.",
    ["model_version", "stage"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

ML_STAGE_REPORTS_TOTAL = Gauge(
    "arhis_ml_stage_reports_total",
    "Total moderation reports scored by model version and stage.",
    ["model_version", "stage"],
)

ML_STAGE_MANUAL_REVIEWS_TOTAL = Gauge(
    "arhis_ml_stage_manual_reviews_total",
    "Manual-reviewed reports by model version and stage.",
    ["model_version", "stage"],
)

ML_STAGE_MANUAL_REVIEW_RATE = Gauge(
    "arhis_ml_stage_manual_review_rate",
    "Manual review rate by model version and stage.",
    ["model_version", "stage"],
)

ML_STAGE_MANUAL_ACCURACY = Gauge(
    "arhis_ml_stage_manual_accuracy",
    "Manual review accuracy by model version and stage.",
    ["model_version", "stage"],
)

ML_STAGE_AVG_CONFIDENCE = Gauge(
    "arhis_ml_stage_avg_confidence",
    "Average ML confidence by model version and stage.",
    ["model_version", "stage"],
)


async def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup_metrics(app: FastAPI) -> None:
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        path = request.url.path
        method = request.method
        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
        REQUESTS_TOTAL.labels(method=method, path=path, status=str(response.status_code)).inc()
        return response

    app.get("/metrics", include_in_schema=False)(metrics_endpoint)


def observe_model_confidence(confidence: float) -> None:
    MODEL_CONFIDENCE.observe(confidence)


def set_queue_depth(value: int) -> None:
    MODERATION_QUEUE_DEPTH.set(value)


def observe_ml_confidence(confidence: float, *, model_version: str | None = None, stage: str | None = None) -> None:
    ML_CONFIDENCE.observe(confidence)
    if model_version and stage:
        ML_CONFIDENCE_BY_MODEL.labels(model_version=model_version, stage=stage).observe(confidence)


def observe_ml_inference_latency(
    seconds: float,
    *,
    model_version: str | None = None,
    stage: str | None = None,
) -> None:
    ML_INFERENCE_LATENCY.observe(max(0.0, seconds))
    if model_version and stage:
        ML_INFERENCE_LATENCY_BY_MODEL.labels(model_version=model_version, stage=stage).observe(max(0.0, seconds))


def increment_ml_routed(route: str) -> None:
    ML_REPORTS_ROUTED_TOTAL.labels(route=route).inc()


def increment_ml_inference(verdict: str, *, model_version: str | None = None, stage: str | None = None) -> None:
    ML_INFERENCE_TOTAL.labels(verdict=verdict).inc()
    if model_version and stage:
        ML_INFERENCE_BY_MODEL_TOTAL.labels(model_version=model_version, stage=stage, verdict=verdict).inc()


def increment_ml_auto_action(action: str, *, model_version: str | None = None, stage: str | None = None) -> None:
    ML_AUTO_ACTION_TOTAL.labels(action=action).inc()
    if model_version and stage:
        ML_AUTO_ACTION_BY_MODEL_TOTAL.labels(model_version=model_version, stage=stage, action=action).inc()


def increment_ml_model_stage(stage: str, *, model_version: str | None = None) -> None:
    ML_MODEL_STAGE_TOTAL.labels(stage=stage).inc()
    if model_version:
        ML_MODEL_STAGE_BY_MODEL_TOTAL.labels(model_version=model_version, stage=stage).inc()


def increment_ml_manual_escalation(reason: str, *, model_version: str | None = None, stage: str | None = None) -> None:
    ML_MANUAL_ESCALATION_TOTAL.labels(reason=reason).inc()
    if model_version and stage:
        ML_MANUAL_ESCALATION_BY_MODEL_TOTAL.labels(model_version=model_version, stage=stage, reason=reason).inc()


def increment_model_registry_validation(result: str) -> None:
    MODEL_REGISTRY_VALIDATION_TOTAL.labels(result=result).inc()


def increment_model_registry_deployment(status: str) -> None:
    MODEL_REGISTRY_DEPLOYMENT_TOTAL.labels(status=status).inc()


def set_data_psi(feature: str, value: float) -> None:
    DATA_PSI.labels(feature=feature).set(value)


def set_data_new_token_share(value: float) -> None:
    DATA_NEW_TOKEN_SHARE.set(value)


def set_data_freshness_seconds(value: float) -> None:
    DATA_FRESHNESS_SECONDS.set(value)


def set_model_sample_manual_accuracy(value: float) -> None:
    MODEL_SAMPLE_MANUAL_ACCURACY.set(value)


def set_model_sample_manual_count(value: int) -> None:
    MODEL_SAMPLE_MANUAL_COUNT.set(value)


def set_retraining_trigger(reason: str, active: bool) -> None:
    RETRAINING_TRIGGER_ACTIVE.labels(reason=reason).set(1 if active else 0)


def set_ml_stage_reports_total(model_version: str, stage: str, value: int) -> None:
    ML_STAGE_REPORTS_TOTAL.labels(model_version=model_version, stage=stage).set(value)


def set_ml_stage_manual_reviews_total(model_version: str, stage: str, value: int) -> None:
    ML_STAGE_MANUAL_REVIEWS_TOTAL.labels(model_version=model_version, stage=stage).set(value)


def set_ml_stage_manual_review_rate(model_version: str, stage: str, value: float) -> None:
    ML_STAGE_MANUAL_REVIEW_RATE.labels(model_version=model_version, stage=stage).set(value)


def set_ml_stage_manual_accuracy(model_version: str, stage: str, value: float) -> None:
    ML_STAGE_MANUAL_ACCURACY.labels(model_version=model_version, stage=stage).set(value)


def set_ml_stage_avg_confidence(model_version: str, stage: str, value: float) -> None:
    ML_STAGE_AVG_CONFIDENCE.labels(model_version=model_version, stage=stage).set(value)


def clear_ml_stage_metrics(model_version: str, stage: str) -> None:
    for metric in (
        ML_STAGE_REPORTS_TOTAL,
        ML_STAGE_MANUAL_REVIEWS_TOTAL,
        ML_STAGE_MANUAL_REVIEW_RATE,
        ML_STAGE_MANUAL_ACCURACY,
        ML_STAGE_AVG_CONFIDENCE,
    ):
        try:
            metric.remove(model_version, stage)
        except KeyError:
            continue


def increment_posts_created() -> None:
    POSTS_CREATED_TOTAL.inc()


def increment_comments_created() -> None:
    COMMENTS_CREATED_TOTAL.inc()


def increment_reports_created() -> None:
    REPORTS_CREATED_TOTAL.inc()


def increment_comments_hidden() -> None:
    COMMENTS_HIDDEN_TOTAL.inc()


def observe_moderation_decision_latency(seconds: float) -> None:
    MODERATION_DECISION_LATENCY_SECONDS.observe(max(0.0, seconds))


async def refresh_users_total_once() -> int:
    from src.app.core.database import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(select(func.count()).select_from(User))
        total = int(result.scalar_one() or 0)
        USERS_TOTAL.set(total)
        return total


async def refresh_users_total_loop(interval_seconds: int = 15) -> None:
    while True:
        try:
            await refresh_users_total_once()
        except Exception:  # pragma: no cover - monitoring should not crash the app
            logger.exception("failed_to_refresh_users_total")
        await asyncio.sleep(interval_seconds)
