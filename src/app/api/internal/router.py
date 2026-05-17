from fastapi import APIRouter

from src.app.api.internal.routes.alerts import router as alerts_router

internal_router = APIRouter()
internal_router.include_router(alerts_router)
