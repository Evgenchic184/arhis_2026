from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.deps import require_admin
from src.app.core.database import get_db_session
from src.app.schemas.alerts import AlertRead
from src.app.services.system_alerts import system_alert_service

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertRead])
async def list_alerts(
    _: object = Depends(require_admin),
    status: str = Query(default="all", pattern="^(all|active|resolved)$"),
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db_session),
) -> list[AlertRead]:
    rows = await system_alert_service.list_alerts(db, status=status, limit=limit)
    return [
        AlertRead(
            id=row.id,
            fingerprint=row.fingerprint,
            status=row.status,
            alertname=row.alertname,
            severity=row.severity,
            summary=row.summary,
            description=row.description,
            labels=row.labels,
            annotations=row.annotations,
            receiver=row.receiver,
            generator_url=row.generator_url,
            starts_at=row.starts_at,
            ends_at=row.ends_at,
            resolved_at=row.resolved_at,
            received_at=row.received_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
            is_active=row.is_active,
        )
        for row in rows
    ]
