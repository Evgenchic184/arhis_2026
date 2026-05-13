from __future__ import annotations

from functools import lru_cache

from src.app.core.settings import get_settings
from src.feature_store.config import FeatureConfig, FeatureConfigStore, feature_config_from_params
from src.utils import read_params


def _settings_default_feature_config() -> FeatureConfig:
    settings = get_settings()
    return FeatureConfig.from_dict(
        {
            "text_column": settings.feature_config_text_column,
            "training_feature_columns": [
                column.strip()
                for column in settings.feature_config_training_feature_columns.split(",")
                if column.strip()
            ],
            "inference_feature_columns": [
                column.strip()
                for column in settings.feature_config_inference_feature_columns.split(",")
                if column.strip()
            ],
            "online_user_feature_columns": [
                column.strip()
                for column in settings.feature_config_online_user_feature_columns.split(",")
                if column.strip()
            ],
            "version": settings.feature_config_version,
        }
    )


def get_default_runtime_feature_config() -> FeatureConfig:
    try:
        params_config = feature_config_from_params(read_params())
        return params_config
    except Exception:
        return _settings_default_feature_config()


@lru_cache(maxsize=8)
def get_feature_config_store(
    redis_url: str | None = None,
    namespace: str | None = None,
) -> FeatureConfigStore:
    settings = get_settings()
    return FeatureConfigStore(
        redis_url=redis_url or settings.redis_url,
        namespace=namespace or settings.feature_store_namespace,
        default_config=get_default_runtime_feature_config(),
    )
