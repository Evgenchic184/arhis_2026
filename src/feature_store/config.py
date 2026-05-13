from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

try:
    import redis.asyncio as redis
except Exception:  # pragma: no cover - fallback when redis isn't installed
    redis = None

from src.feature_store.feature_sets import (
    BASE_FEATURE_COLUMNS,
    BASE_TEXT_COLUMN,
    BASE_USER_FEATURE_COLUMNS,
    validate_model_feature_columns,
    validate_user_feature_columns,
)

CONFIG_KEY = "feature_config"


@dataclass(slots=True)
class FeatureConfig:
    text_column: str = BASE_TEXT_COLUMN
    training_feature_columns: list[str] = field(default_factory=lambda: list(BASE_FEATURE_COLUMNS))
    inference_feature_columns: list[str] = field(default_factory=lambda: list(BASE_FEATURE_COLUMNS))
    online_user_feature_columns: list[str] = field(default_factory=lambda: list(BASE_USER_FEATURE_COLUMNS))
    version: int = 1
    updated_at: str | None = None

    def validate(self) -> None:
        self.training_feature_columns = validate_model_feature_columns(self.training_feature_columns)
        self.inference_feature_columns = validate_model_feature_columns(self.inference_feature_columns)
        self.online_user_feature_columns = validate_user_feature_columns(self.online_user_feature_columns)

        if self.text_column not in {BASE_TEXT_COLUMN}:
            raise ValueError(f"Unsupported text column: {self.text_column}")

    def bump_version(self) -> None:
        self.version += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["updated_at"] = self.updated_at or datetime.now(timezone.utc).isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureConfig":
        config = cls(
            text_column=payload.get("text_column", BASE_TEXT_COLUMN),
            training_feature_columns=list(payload.get("training_feature_columns", BASE_FEATURE_COLUMNS)),
            inference_feature_columns=list(payload.get("inference_feature_columns", BASE_FEATURE_COLUMNS)),
            online_user_feature_columns=list(payload.get("online_user_feature_columns", BASE_USER_FEATURE_COLUMNS)),
            version=int(payload.get("version", 1)),
            updated_at=payload.get("updated_at"),
        )
        config.validate()
        return config


class FeatureConfigStore:
    def __init__(
        self,
        redis_url: str | None = None,
        namespace: str = "arhis",
        default_config: FeatureConfig | None = None,
    ) -> None:
        self.redis_url = redis_url
        self.namespace = namespace
        self._memory_config = default_config or FeatureConfig()
        self._client = redis.from_url(redis_url, decode_responses=True) if redis and redis_url else None

    @property
    def redis_key(self) -> str:
        return f"{self.namespace}:{CONFIG_KEY}"

    async def get(self) -> FeatureConfig:
        if self._client is None:
            return self._memory_config

        try:
            raw = await self._client.get(self.redis_key)
            if not raw:
                return self._memory_config
            return FeatureConfig.from_dict(json.loads(raw))
        except Exception:
            return self._memory_config

    async def set(self, config: FeatureConfig) -> FeatureConfig:
        config.validate()
        config.bump_version()
        if self._client is None:
            self._memory_config = config
            return config

        try:
            await self._client.set(self.redis_key, json.dumps(config.to_dict(), ensure_ascii=False))
            return config
        except Exception:
            self._memory_config = config
            return config


def feature_config_from_params(params: dict[str, Any] | None = None) -> FeatureConfig:
    params = params or {}
    feature_params = params.get("features", {})
    runtime_params = feature_params.get("runtime", feature_params)
    config = FeatureConfig(
        text_column=runtime_params.get("text_column", BASE_TEXT_COLUMN),
        training_feature_columns=list(runtime_params.get("training_feature_columns", BASE_FEATURE_COLUMNS)),
        inference_feature_columns=list(runtime_params.get("inference_feature_columns", BASE_FEATURE_COLUMNS)),
        online_user_feature_columns=list(
            runtime_params.get("online_user_feature_columns", BASE_USER_FEATURE_COLUMNS)
        ),
        version=int(runtime_params.get("version", feature_params.get("version", 1))),
        updated_at=runtime_params.get("updated_at", feature_params.get("updated_at")),
    )
    config.validate()
    return config


def feature_config_from_settings(
    *,
    text_column: str = BASE_TEXT_COLUMN,
    training_feature_columns: Iterable[str] = BASE_FEATURE_COLUMNS,
    inference_feature_columns: Iterable[str] = BASE_FEATURE_COLUMNS,
    online_user_feature_columns: Iterable[str] = BASE_USER_FEATURE_COLUMNS,
    version: int = 1,
) -> FeatureConfig:
    config = FeatureConfig(
        text_column=text_column,
        training_feature_columns=list(training_feature_columns),
        inference_feature_columns=list(inference_feature_columns),
        online_user_feature_columns=list(online_user_feature_columns),
        version=version,
    )
    config.validate()
    return config


async def load_runtime_feature_config(
    redis_url: str | None = None,
    namespace: str = "arhis",
    params: dict[str, Any] | None = None,
) -> FeatureConfig:
    store = FeatureConfigStore(
        redis_url=redis_url,
        namespace=namespace,
        default_config=feature_config_from_params(params),
    )
    return await store.get()


@lru_cache(maxsize=1)
def get_default_feature_config() -> FeatureConfig:
    return FeatureConfig()
