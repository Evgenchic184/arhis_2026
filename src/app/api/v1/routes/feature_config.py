from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from src.app.api.deps import RequestUserContext, require_moderator
from src.app.core.feature_config import get_feature_config_store
from src.app.schemas.feature_config import FeatureConfigRead, FeatureConfigUpdate
from src.feature_store.config import FeatureConfig
from src.feature_store.feature_sets import (
    AVAILABLE_MODEL_FEATURE_COLUMNS,
    AVAILABLE_TEXT_FEATURE_COLUMNS,
    AVAILABLE_USER_FEATURE_COLUMNS,
)

router = APIRouter(prefix="/config/features", tags=["feature-config"])
logger = logging.getLogger(__name__)


def _to_read_model(config: FeatureConfig) -> FeatureConfigRead:
    return FeatureConfigRead(
        text_column=config.text_column,
        training_feature_columns=list(config.training_feature_columns),
        inference_feature_columns=list(config.inference_feature_columns),
        online_user_feature_columns=list(config.online_user_feature_columns),
        version=config.version,
        updated_at=config.updated_at,
        available_model_feature_columns=list(AVAILABLE_MODEL_FEATURE_COLUMNS),
        available_user_feature_columns=list(AVAILABLE_USER_FEATURE_COLUMNS),
        available_text_feature_columns=list(AVAILABLE_TEXT_FEATURE_COLUMNS),
    )


@router.get("", response_model=FeatureConfigRead)
async def read_feature_config(
    _: RequestUserContext = Depends(require_moderator),
) -> FeatureConfigRead:
    config = await get_feature_config_store().get()
    return _to_read_model(config)


@router.patch("", response_model=FeatureConfigRead)
async def update_feature_config(
    payload: FeatureConfigUpdate,
    _: RequestUserContext = Depends(require_moderator),
) -> FeatureConfigRead:
    store = get_feature_config_store()
    current = await store.get()
    updated = FeatureConfig(
        text_column=payload.text_column if payload.text_column is not None else current.text_column,
        training_feature_columns=(
            payload.training_feature_columns if payload.training_feature_columns is not None else list(current.training_feature_columns)
        ),
        inference_feature_columns=(
            payload.inference_feature_columns if payload.inference_feature_columns is not None else list(current.inference_feature_columns)
        ),
        online_user_feature_columns=(
            payload.online_user_feature_columns if payload.online_user_feature_columns is not None else list(current.online_user_feature_columns)
        ),
        version=current.version,
        updated_at=current.updated_at,
    )
    updated = await store.set(updated)
    logger.info(
        "feature_config_updated",
        extra={
            "event": "feature_config_updated",
            "version": updated.version,
            "text_column": updated.text_column,
            "training_feature_columns": updated.training_feature_columns,
            "inference_feature_columns": updated.inference_feature_columns,
            "online_user_feature_columns": updated.online_user_feature_columns,
        },
    )
    return _to_read_model(updated)
