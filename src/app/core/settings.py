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
    kafka_bootstrap_servers: str = Field(default="")
    kafka_domain_events_topic: str = Field(default="arhis.domain.events")
    kafka_moderation_ml_requests_topic: str = Field(default="arhis.moderation.ml.requests")
    s3_endpoint_url: str = Field(default="")
    s3_access_key_id: str = Field(default="")
    s3_secret_access_key: str = Field(default="")
    s3_bucket_name: str = Field(default="arhis-event-logs")
    retraining_dataset_bucket_name: str = Field(default="arhis-retraining-datasets")
    s3_region_name: str = Field(default="us-east-1")
    s3_use_ssl: bool = Field(default=False)
    prometheus_base_url: str = Field(default="http://localhost:9090")
    parquet_sink_prefix: str = Field(default="event_logs")
    parquet_sink_flush_rows: int = Field(default=100)
    parquet_sink_flush_seconds: int = Field(default=10)
    moderation_ml_route_rate: float = Field(default=0.8)
    moderation_ml_manual_sample_rate: float = Field(default=0.1)
    moderation_ml_confidence_threshold_low: float = Field(default=0.65)
    moderation_ml_confidence_threshold_high: float = Field(default=0.9)
    ml_model_path: str = Field(default="models/cyberbullying_moderation.joblib")
    ml_model_version: str = Field(default="dev")
    model_registry_enabled: bool = Field(default=True)
    model_registry_bucket_name: str = Field(default="arhis-model-registry")
    model_registry_cache_dir: str = Field(default="models/cache")
    model_registry_model_name: str = Field(default="cyberbullying_moderation")
    model_registry_canary_traffic_percent: int = Field(default=10)
    model_registry_validation_sample_size: int = Field(default=32)
    model_registry_required_accuracy: float = Field(default=1.0)
    model_registry_auto_promote_enabled: bool = Field(default=True)
    model_registry_auto_promote_check_interval_seconds: int = Field(default=60)
    model_registry_auto_promote_min_age_seconds: int = Field(default=300)
    model_registry_auto_promote_min_reports: int = Field(default=100)
    model_registry_auto_promote_min_manual_samples: int = Field(default=5)
    ml_worker_metrics_port: int = Field(default=8011)
    retraining_enabled: bool = Field(default=True)
    retraining_cooldown_seconds: int = Field(default=86400)
    retraining_monitor_interval_seconds: int = Field(default=600)
    retraining_data_window_hours: int = Field(default=168)
    retraining_manual_sample_window_hours: int = Field(default=168)
    retraining_psi_threshold: float = Field(default=0.2)
    retraining_new_token_share_threshold: float = Field(default=0.1)
    retraining_manual_accuracy_threshold: float = Field(default=0.95)
    retraining_manual_accuracy_drop_threshold: float = Field(default=0.05)
    retraining_manual_accuracy_min_samples: int = Field(default=20)
    retraining_monitor_metrics_port: int = Field(default=8012)
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

    @field_validator(
        "debug",
        "sqlalchemy_echo",
        "metrics_enabled",
        "model_registry_enabled",
        "retraining_enabled",
        mode="before",
    )
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

    @field_validator(
        "model_registry_canary_traffic_percent",
        "model_registry_validation_sample_size",
        "model_registry_auto_promote_check_interval_seconds",
        "model_registry_auto_promote_min_age_seconds",
        "model_registry_auto_promote_min_reports",
        "model_registry_auto_promote_min_manual_samples",
        "ml_worker_metrics_port",
        "retraining_cooldown_seconds",
        "retraining_monitor_interval_seconds",
        "retraining_data_window_hours",
        "retraining_manual_sample_window_hours",
        "retraining_manual_accuracy_min_samples",
        "retraining_monitor_metrics_port",
        mode="before",
    )
    @classmethod
    def parse_int_flags(cls, value: object) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            return int(value)
        return int(value)

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "")

    @property
    def cors_allow_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
