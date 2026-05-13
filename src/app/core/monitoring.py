from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

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

MODEL_CONFIDENCE = Histogram(
    "arhis_model_confidence",
    "Confidence values returned by the ML model.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0),
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
