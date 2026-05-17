from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.app.api.v1.router import api_router
from src.app.api.internal.router import internal_router
from src.app.core.database import engine
from src.app.core.logging import configure_logging
from src.app.core.monitoring import refresh_users_total_loop, setup_metrics
from src.app.core.request_context import request_id_var
from src.app.core.settings import get_settings

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_: FastAPI):
    metrics_task = asyncio.create_task(refresh_users_total_loop())
    yield
    metrics_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await metrics_task
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            request_id_var.reset(token)


app.add_middleware(RequestIdMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)
app.include_router(internal_router)
if settings.metrics_enabled:
    setup_metrics(app)


@app.get("/health", tags=["system"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
