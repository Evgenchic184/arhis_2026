from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.api.deps import RequestUserContext, require_admin
from src.app.core.database import get_db_session
from src.app.core.settings import get_settings
from src.app.models.enums import ModelVersionStatus
from src.app.models.model_registry import ModelVersion
from src.app.schemas.model_registry import ModelRegistryOverviewRead, ModelVersionRead
from src.app.services.model_registry import ModelRegistryService

router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger(__name__)


def _serialize_model_version(row: ModelVersion) -> ModelVersionRead:
    return ModelVersionRead.model_validate(row)


@router.get("", response_model=ModelRegistryOverviewRead)
async def list_model_versions(
    _: RequestUserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
) -> ModelRegistryOverviewRead:
    settings = get_settings()
    try:
        result = await db.execute(
            select(ModelVersion).where(ModelVersion.model_name == settings.model_registry_model_name).order_by(
                ModelVersion.created_at.desc()
            )
        )
    except (ProgrammingError, SQLAlchemyError):
        logger.info("model_registry_not_ready", extra={"event": "model_registry_not_ready"})
        return ModelRegistryOverviewRead(
            model_name=settings.model_registry_model_name,
            active_production=None,
            active_canary=None,
            versions=[],
        )

    versions = list(result.scalars().all())
    production = next((row for row in versions if row.status == ModelVersionStatus.production), None)
    canary = next((row for row in versions if row.status == ModelVersionStatus.canary), None)
    return ModelRegistryOverviewRead(
        model_name=settings.model_registry_model_name,
        active_production=_serialize_model_version(production) if production else None,
        active_canary=_serialize_model_version(canary) if canary else None,
        versions=[_serialize_model_version(row) for row in versions],
    )


@router.post("/{version}/promote", response_model=ModelVersionRead)
async def promote_model_version(
    version: str,
    _: RequestUserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
) -> ModelVersionRead:
    settings = get_settings()
    service = ModelRegistryService(settings)
    try:
        row = await service.promote_version(db, settings.model_registry_model_name, version)
        await db.commit()
    except (ProgrammingError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry is not ready.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _serialize_model_version(row)


@router.post("/rollback", response_model=ModelVersionRead)
async def rollback_model(
    _: RequestUserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
) -> ModelVersionRead:
    settings = get_settings()
    service = ModelRegistryService(settings)
    try:
        row = await service.rollback_latest(db, settings.model_registry_model_name)
        await db.commit()
    except (ProgrammingError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model registry is not ready.",
        ) from exc
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No production model found.")
    return _serialize_model_version(row)
