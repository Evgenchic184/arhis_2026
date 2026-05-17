from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config

from src.app.core.settings import Settings, get_settings


@dataclass(slots=True)
class EventArchiveQuery:
    bucket_name: str
    prefix: str = "event_logs"
    event_types: list[str] | None = None
    days_back: int | None = None


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


class EventArchiveStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = _build_s3_client(self.settings)

    def _list_objects_sync(self, query: EventArchiveQuery) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=query.bucket_name, Prefix=query.prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith(".parquet"):
                    continue
                keys.append(key)
        return sorted(keys)

    def _read_object_sync(self, bucket_name: str, key: str) -> pd.DataFrame:
        response = self._client.get_object(Bucket=bucket_name, Key=key)
        data = response["Body"].read()
        return pd.read_parquet(BytesIO(data))

    async def read_events(self, query: EventArchiveQuery) -> pd.DataFrame:
        keys = await asyncio.to_thread(self._list_objects_sync, query)
        if not keys:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for key in keys:
            frame = await asyncio.to_thread(self._read_object_sync, query.bucket_name, key)
            if frame.empty:
                continue
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        frame = pd.concat(frames, ignore_index=True)
        if query.event_types:
            frame = frame[frame["event_type"].isin(query.event_types)].copy()
        if query.days_back is not None and "created_at" in frame.columns:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=query.days_back)
            frame["created_at_ts"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
            frame = frame[frame["created_at_ts"] >= cutoff].copy()
        return frame
