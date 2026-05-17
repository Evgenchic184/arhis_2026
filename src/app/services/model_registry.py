from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import boto3
import joblib
import pandas as pd
import numpy as np
from botocore.config import Config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.core.monitoring import increment_model_registry_deployment, increment_model_registry_validation
from src.app.core.settings import Settings, get_settings
from src.app.models.enums import ModelVersionStatus
from src.app.models.model_registry import ModelVersion
from src.feature_store.feature_sets import BASE_LABEL_COLUMN

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelSnapshot:
    id: UUID
    model_name: str
    version: str
    status: ModelVersionStatus
    artifact_uri: str
    metadata_uri: str | None
    feature_config_version: int
    traffic_percent: int
    required_validation_accuracy: float
    validation_sample_size: int
    validation_accuracy: float | None
    validation_metrics: dict[str, Any] | None
    training_metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    active_from_at: datetime | None
    promoted_at: datetime | None
    rolled_back_at: datetime | None


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Unsupported artifact URI: {uri}")
    bucket_and_key = uri.removeprefix("s3://")
    bucket, key = bucket_and_key.split("/", 1)
    return bucket, key


def _build_s3_client(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        aws_access_key_id=settings.s3_access_key_id or None,
        aws_secret_access_key=settings.s3_secret_access_key or None,
        region_name=settings.s3_region_name or None,
        use_ssl=bool(settings.s3_use_ssl),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _ensure_bucket_exists(s3_client, bucket_name: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception:
        s3_client.create_bucket(Bucket=bucket_name)


def _safe_metrics(frame: pd.DataFrame, model) -> dict[str, float]:
    y_true = frame[BASE_LABEL_COLUMN]
    y_pred = model.predict(frame)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(frame)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = 0.0
    return metrics


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


def _json_safe_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return _json_safe_value(payload)


class ModelRegistryService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._s3_client = None
        self._artifact_cache_dir = Path(self.settings.model_registry_cache_dir)
        self._artifact_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return bool(
            self.settings.model_registry_enabled
            and self.settings.s3_endpoint_url
            and self.settings.s3_access_key_id
            and self.settings.s3_secret_access_key
        )

    def _client(self):
        if self._s3_client is None:
            self._s3_client = _build_s3_client(self.settings)
        return self._s3_client

    async def ensure_bucket(self) -> None:
        if not self.enabled:
            return
        await asyncio.to_thread(_ensure_bucket_exists, self._client(), self.settings.model_registry_bucket_name)

    async def upload_artifacts(
        self,
        *,
        model_name: str,
        version: str,
        local_model_path: Path,
        metadata_path: Path | None = None,
        validation_report_path: Path | None = None,
    ) -> tuple[str, str | None]:
        if not self.enabled:
            raise RuntimeError("Model registry storage is not configured.")

        await self.ensure_bucket()
        bucket = self.settings.model_registry_bucket_name
        base_key = f"models/{model_name}/{version}"
        artifact_key = f"{base_key}/model.joblib"
        metadata_key = f"{base_key}/metadata.json"

        async def _put(path: Path, key: str) -> None:
            await asyncio.to_thread(
                self._client().upload_file,
                str(path),
                bucket,
                key,
            )

        await _put(local_model_path, artifact_key)
        if metadata_path and metadata_path.exists():
            await _put(metadata_path, metadata_key)
        if validation_report_path and validation_report_path.exists():
            await _put(validation_report_path, f"{base_key}/validation.json")

        artifact_uri = f"s3://{bucket}/{artifact_key}"
        metadata_uri = f"s3://{bucket}/{metadata_key}" if metadata_path and metadata_path.exists() else None
        return artifact_uri, metadata_uri

    async def download_artifact(self, artifact_uri: str) -> Path:
        bucket, key = _parse_s3_uri(artifact_uri)
        cache_key = key.replace("/", "__")
        local_path = self._artifact_cache_dir / cache_key
        if local_path.exists():
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._client().download_file, bucket, key, str(local_path))
        return local_path

    async def load_model(self, artifact_uri: str):
        local_path = await self.download_artifact(artifact_uri)
        return await asyncio.to_thread(joblib.load, local_path)

    async def list_active_versions(self, db: AsyncSession, model_name: str) -> list[ModelSnapshot]:
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.status.in_([ModelVersionStatus.production, ModelVersionStatus.canary]))
            .order_by(ModelVersion.created_at.asc())
        )
        result = await db.execute(stmt)
        snapshots: list[ModelSnapshot] = []
        for row in result.scalars().all():
            snapshots.append(
                ModelSnapshot(
                    id=row.id,
                    model_name=row.model_name,
                    version=row.version,
                    status=row.status,
                    artifact_uri=row.artifact_uri,
                    metadata_uri=row.metadata_uri,
                    feature_config_version=row.feature_config_version,
                    traffic_percent=row.traffic_percent,
                    required_validation_accuracy=row.required_validation_accuracy,
                    validation_sample_size=row.validation_sample_size,
                    validation_accuracy=row.validation_accuracy,
                    validation_metrics=row.validation_metrics,
                    training_metadata=row.training_metadata,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    active_from_at=row.active_from_at,
                    promoted_at=row.promoted_at,
                    rolled_back_at=row.rolled_back_at,
                )
            )
        return snapshots

    async def validate_candidate(
        self,
        model,
        validation_frame: pd.DataFrame,
        *,
        sample_size: int | None = None,
        required_accuracy: float | None = None,
    ) -> tuple[bool, dict[str, float], pd.DataFrame]:
        required_accuracy = self.settings.model_registry_required_accuracy if required_accuracy is None else required_accuracy
        if validation_frame.empty:
            raise ValueError("Validation frame is empty.")
        sample_size = sample_size or self.settings.model_registry_validation_sample_size
        sample_size = min(sample_size, len(validation_frame))
        sample_frame = validation_frame.sample(n=sample_size, random_state=42).copy()
        metrics = _safe_metrics(sample_frame, model)
        passed = metrics["accuracy"] >= required_accuracy
        return passed, metrics, sample_frame

    async def register_version(
        self,
        db: AsyncSession,
        *,
        model_name: str,
        version: str,
        local_model_path: Path,
        metadata_path: Path | None,
        validation_report_path: Path | None,
        feature_config_version: int,
        validation_accuracy: float,
        validation_sample_size: int,
        validation_metrics: dict[str, Any],
        training_metadata: dict[str, Any] | None,
        canary_traffic_percent: int | None = None,
    ) -> ModelVersion:
        increment_model_registry_validation("pass")
        artifact_uri, metadata_uri = await self.upload_artifacts(
            model_name=model_name,
            version=version,
            local_model_path=local_model_path,
            metadata_path=metadata_path,
            validation_report_path=validation_report_path,
        )

        current_production = await self._get_current_by_status(db, model_name, ModelVersionStatus.production)
        existing_canary = await self._get_current_by_status(db, model_name, ModelVersionStatus.canary)
        now = datetime.now(timezone.utc)

        if existing_canary is not None:
            existing_canary.status = ModelVersionStatus.archived
            existing_canary.traffic_percent = 0
            existing_canary.updated_at = now

        status = ModelVersionStatus.production if current_production is None else ModelVersionStatus.canary
        traffic_percent = 100 if status == ModelVersionStatus.production else int(
            canary_traffic_percent or self.settings.model_registry_canary_traffic_percent
        )
        if status == ModelVersionStatus.production:
            traffic_percent = 100

        row = ModelVersion(
            model_name=model_name,
            version=version,
            status=status,
            artifact_uri=artifact_uri,
            metadata_uri=metadata_uri,
            feature_config_version=feature_config_version,
            traffic_percent=traffic_percent,
            required_validation_accuracy=self.settings.model_registry_required_accuracy,
            validation_sample_size=validation_sample_size,
            validation_accuracy=validation_accuracy,
            validation_metrics=_json_safe_payload(validation_metrics),
            training_metadata=_json_safe_payload(training_metadata),
            active_from_at=now,
            promoted_at=now if status in {ModelVersionStatus.production, ModelVersionStatus.canary} else None,
        )
        db.add(row)
        await db.flush()
        increment_model_registry_deployment(status.value)
        return row

    async def mark_rejected(
        self,
        db: AsyncSession,
        *,
        model_name: str,
        version: str,
        local_model_path: Path,
        metadata_path: Path | None,
        validation_report_path: Path | None,
        feature_config_version: int,
        validation_accuracy: float,
        validation_sample_size: int,
        validation_metrics: dict[str, Any],
        training_metadata: dict[str, Any] | None,
        notes: str | None = None,
    ) -> ModelVersion:
        artifact_uri, metadata_uri = await self.upload_artifacts(
            model_name=model_name,
            version=version,
            local_model_path=local_model_path,
            metadata_path=metadata_path,
            validation_report_path=validation_report_path,
        )
        now = datetime.now(timezone.utc)
        row = ModelVersion(
            model_name=model_name,
            version=version,
            status=ModelVersionStatus.rejected,
            artifact_uri=artifact_uri,
            metadata_uri=metadata_uri,
            feature_config_version=feature_config_version,
            traffic_percent=0,
            required_validation_accuracy=self.settings.model_registry_required_accuracy,
            validation_sample_size=validation_sample_size,
            validation_accuracy=validation_accuracy,
            validation_metrics=_json_safe_payload(validation_metrics),
            training_metadata=_json_safe_payload(training_metadata),
            notes=notes,
            active_from_at=now,
        )
        db.add(row)
        await db.flush()
        increment_model_registry_validation("fail")
        increment_model_registry_deployment("rejected")
        return row

    async def mark_rolled_back(self, db: AsyncSession, model_id: UUID) -> None:
        stmt = select(ModelVersion).where(ModelVersion.id == model_id)
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return
        row.status = ModelVersionStatus.rolled_back
        row.traffic_percent = 0
        row.rolled_back_at = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)
        increment_model_registry_deployment("rolled_back")

    async def promote_version(self, db: AsyncSession, model_name: str, version: str) -> ModelVersion:
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.version == version)
        )
        result = await db.execute(stmt)
        target = result.scalar_one_or_none()
        if target is None:
            raise ValueError(f"Model version not found: {model_name}:{version}")
        if target.status not in {ModelVersionStatus.canary, ModelVersionStatus.validated, ModelVersionStatus.candidate}:
            raise ValueError(f"Model version is not promotable: {target.status.value}")

        current_production = await self._get_current_by_status(db, model_name, ModelVersionStatus.production)
        now = datetime.now(timezone.utc)
        if current_production is not None and current_production.id != target.id:
            current_production.status = ModelVersionStatus.archived
            current_production.traffic_percent = 0
            current_production.updated_at = now

        target.status = ModelVersionStatus.production
        target.traffic_percent = 100
        target.promoted_at = now
        target.active_from_at = now
        target.updated_at = now
        increment_model_registry_deployment("promoted")
        return target

    async def rollback_latest(self, db: AsyncSession, model_name: str) -> ModelVersion | None:
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.status == ModelVersionStatus.production)
            .order_by(ModelVersion.created_at.desc())
        )
        result = await db.execute(stmt)
        current_production = result.scalar_one_or_none()
        if current_production is None:
            return None

        candidate_stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.status.in_([ModelVersionStatus.archived, ModelVersionStatus.rolled_back]))
            .order_by(ModelVersion.created_at.desc())
        )
        candidate_result = await db.execute(candidate_stmt)
        previous = candidate_result.scalars().first()
        if previous is None:
            return None

        now = datetime.now(timezone.utc)
        current_production.status = ModelVersionStatus.rolled_back
        current_production.traffic_percent = 0
        current_production.rolled_back_at = now
        current_production.updated_at = now

        previous.status = ModelVersionStatus.production
        previous.traffic_percent = 100
        previous.promoted_at = now
        previous.active_from_at = now
        previous.updated_at = now
        increment_model_registry_deployment("rolled_back")
        return previous

    async def _get_current_by_status(
        self,
        db: AsyncSession,
        model_name: str,
        status: ModelVersionStatus,
    ) -> ModelVersion | None:
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.status == status)
            .order_by(ModelVersion.created_at.desc())
        )
        result = await db.execute(stmt)
        return result.scalars().first()

    def route_stage_for_report(self, report_id: UUID, canary_traffic_percent: int | None) -> str:
        traffic_percent = int(canary_traffic_percent or self.settings.model_registry_canary_traffic_percent)
        if traffic_percent <= 0:
            return "production"
        bucket = report_id.int % 100
        return "canary" if bucket < traffic_percent else "production"


def score_validation_sample(model, validation_frame: pd.DataFrame) -> dict[str, float]:
    return _safe_metrics(validation_frame, model)
