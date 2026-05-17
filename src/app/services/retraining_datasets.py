from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from src.app.core.settings import Settings, get_settings


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


@dataclass(slots=True)
class RetrainingDatasetBundle:
    version: str
    dataset_path: Path
    train_path: Path
    val_path: Path
    test_path: Path
    baseline_vocab_path: Path
    summary_path: Path
    manifest_path: Path


class RetrainingDatasetArchiveService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = _build_s3_client(self.settings)

    async def ensure_bucket(self) -> None:
        await asyncio.to_thread(_ensure_bucket_exists, self._client, self.settings.retraining_dataset_bucket_name)

    async def upload_bundle(self, bundle: RetrainingDatasetBundle) -> dict[str, str]:
        await self.ensure_bucket()
        bucket = self.settings.retraining_dataset_bucket_name
        base_key = f"models/{self.settings.model_registry_model_name}/{bundle.version}"

        uploads: dict[str, str] = {
            "dataset": f"s3://{bucket}/{base_key}/retraining_dataset.parquet",
            "train": f"s3://{bucket}/{base_key}/train.parquet",
            "val": f"s3://{bucket}/{base_key}/val.parquet",
            "test": f"s3://{bucket}/{base_key}/test.parquet",
            "baseline_vocab": f"s3://{bucket}/{base_key}/baseline_vocab.json",
            "summary": f"s3://{bucket}/{base_key}/dataset_summary.json",
            "manifest": f"s3://{bucket}/{base_key}/manifest.json",
        }

        def _put(local_path: Path, key: str) -> None:
            self._client.upload_file(str(local_path), bucket, key)

        files: dict[str, Path] = {
            "dataset": bundle.dataset_path,
            "train": bundle.train_path,
            "val": bundle.val_path,
            "test": bundle.test_path,
            "baseline_vocab": bundle.baseline_vocab_path,
            "summary": bundle.summary_path,
            "manifest": bundle.manifest_path,
        }

        for name, local_path in files.items():
            await asyncio.to_thread(_put, local_path, uploads[name].removeprefix(f"s3://{bucket}/"))

        return uploads

    async def upload_snapshot(
        self,
        *,
        version: str,
        dataset_path: Path,
        summary_path: Path,
        manifest_path: Path,
    ) -> dict[str, str]:
        await self.ensure_bucket()
        bucket = self.settings.retraining_dataset_bucket_name
        base_key = f"models/{self.settings.model_registry_model_name}/{version}"

        uploads: dict[str, str] = {
            "dataset": f"s3://{bucket}/{base_key}/retraining_dataset.parquet",
            "summary": f"s3://{bucket}/{base_key}/dataset_summary.json",
            "manifest": f"s3://{bucket}/{base_key}/manifest.json",
        }

        def _put(local_path: Path, key: str) -> None:
            self._client.upload_file(str(local_path), bucket, key)

        files: dict[str, Path] = {
            "dataset": dataset_path,
            "summary": summary_path,
            "manifest": manifest_path,
        }

        for name, local_path in files.items():
            await asyncio.to_thread(_put, local_path, uploads[name].removeprefix(f"s3://{bucket}/"))

        return uploads
