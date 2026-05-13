from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Arhis API")
    api_v1_prefix: str = Field(default="/api/v1")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    sqlalchemy_echo: bool = Field(default=False)
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/arhis"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    feature_store_namespace: str = Field(default="arhis")
    metrics_enabled: bool = Field(default=True)
    cors_allow_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")
    cors_allow_origin_regex: str = Field(
        default=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(:\d+)?$"
    )
    feature_config_text_column: str = Field(default="text_prepared")
    feature_config_training_feature_columns: str = Field(
        default="text_length,caps_ratio,has_url,has_mention,is_new_user,reputation_score,reports_last_24h,account_age_days"
    )
    feature_config_inference_feature_columns: str = Field(
        default="text_length,caps_ratio,has_url,has_mention,is_new_user,reputation_score,reports_last_24h,account_age_days"
    )
    feature_config_online_user_feature_columns: str = Field(
        default="is_new_user,reputation_score,reports_last_24h,account_age_days"
    )
    feature_config_version: int = Field(default=1)
    jwt_secret_key: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=60 * 24 * 7)

    @field_validator("debug", "sqlalchemy_echo", "metrics_enabled", mode="before")
    @classmethod
    def parse_bool_flags(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on", "debug"}:
                return True
            if lowered in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return bool(value)

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "")

    @property
    def cors_allow_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
