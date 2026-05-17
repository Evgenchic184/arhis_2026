from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.core.database import get_db_session
from src.app.services.system_alerts import system_alert_service

router = APIRouter(prefix="/internal/alerts", tags=["internal"])


@router.post("/webhook", include_in_schema=False)
async def alertmanager_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, int | str]:
    payload = await request.json()
    ingested = await system_alert_service.ingest_webhook(db, payload if isinstance(payload, dict) else {})
    await db.commit()
    return {"status": "ok", "ingested": ingested}
