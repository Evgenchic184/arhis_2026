from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config

try:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.structs import TopicPartition
except Exception:  # pragma: no cover - optional dependency in dev environments
    AIOKafkaConsumer = None
    TopicPartition = None

from src.app.core.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PartitionBuffer:
    rows: list[dict[str, Any]] = field(default_factory=list)
    next_offset: int | None = None
    first_created_at: str | None = None
    first_seen_monotonic: float = 0.0


def _parse_event_date(created_at: str | None) -> str:
    if not created_at:
        return datetime.now(timezone.utc).date().isoformat()
    parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return parsed.date().isoformat()


def _normalize_message(message) -> dict[str, Any]:
    payload = message.value or {}
    payload_body = payload.get("payload") or {}
    flat = {
        "event_id": payload.get("event_id"),
        "event_type": payload.get("event_type"),
        "aggregate_type": payload.get("aggregate_type"),
        "aggregate_id": payload.get("aggregate_id"),
        "actor_id": payload.get("actor_id"),
        "actor_role": payload.get("actor_role"),
        "request_id": payload.get("request_id"),
        "created_at": payload.get("created_at"),
        "topic": message.topic,
        "partition": message.partition,
        "offset": message.offset,
        "kafka_timestamp_ms": message.timestamp,
        "payload_json": json.dumps(payload_body, ensure_ascii=False, default=str),
    }
    for key, value in payload_body.items():
        flat[f"payload_{key}"] = value
    return flat


def _build_s3_client(settings):
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


async def _flush_partition(
    *,
    consumer: AIOKafkaConsumer,
    s3_client,
    bucket_name: str,
    prefix: str,
    tp,
    state: PartitionBuffer,
) -> None:
    if not state.rows:
        return

    rows = pd.DataFrame(state.rows)
    table = pa.Table.from_pandas(rows, preserve_index=False)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    buffer.seek(0)

    event_date = _parse_event_date(state.first_created_at)
    first_offset = int(state.rows[0]["offset"])
    last_offset = int(state.rows[-1]["offset"])
    topic_name = str(tp.topic).replace("/", "_")
    object_key = (
        f"{prefix}/event_date={event_date}/"
        f"topic={topic_name}/partition={tp.partition}/"
        f"offsets={first_offset:020d}-{last_offset:020d}.parquet"
    )

    await asyncio.to_thread(
        s3_client.put_object,
        Bucket=bucket_name,
        Key=object_key,
        Body=buffer.getvalue(),
        ContentType="application/x-parquet",
    )
    await consumer.commit({tp: state.next_offset})
    logger.info(
        "parquet_sink_flushed",
        extra={
            "event": "parquet_sink_flushed",
            "topic": tp.topic,
            "partition": tp.partition,
            "offset_start": first_offset,
            "offset_end": last_offset,
            "object_key": object_key,
        },
    )


async def consume_to_parquet() -> None:
    settings = get_settings()
    if not settings.kafka_bootstrap_servers or AIOKafkaConsumer is None or TopicPartition is None:
        raise RuntimeError("Kafka consumer is not configured.")

    consumer_topics = [settings.kafka_domain_events_topic]
    if settings.kafka_moderation_ml_requests_topic not in consumer_topics:
        consumer_topics.append(settings.kafka_moderation_ml_requests_topic)

    consumer = AIOKafkaConsumer(
        *consumer_topics,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="arhis-parquet-sink",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        key_deserializer=lambda value: value.decode("utf-8") if value else None,
    )

    s3_client = _build_s3_client(settings)
    buffers: dict[TopicPartition, PartitionBuffer] = defaultdict(PartitionBuffer)
    loop = asyncio.get_running_loop()

    while True:
        try:
            await asyncio.to_thread(_ensure_bucket_exists, s3_client, settings.s3_bucket_name)
            await consumer.start()
            break
        except Exception as exc:
            logger.info(
                "parquet_sink_waiting_for_dependencies",
                extra={"event": "parquet_sink_waiting_for_dependencies", "error": str(exc)},
            )
            await asyncio.sleep(5)

    async def flush_due(force_all: bool = False) -> None:
        ready_partitions: list[TopicPartition] = []
        now = loop.time()
        for tp, state in buffers.items():
            if not state.rows:
                continue
            if force_all or len(state.rows) >= settings.parquet_sink_flush_rows:
                ready_partitions.append(tp)
                continue
            if state.first_seen_monotonic and now - state.first_seen_monotonic >= settings.parquet_sink_flush_seconds:
                ready_partitions.append(tp)

        for tp in ready_partitions:
            state = buffers[tp]
            await _flush_partition(
                consumer=consumer,
                s3_client=s3_client,
                bucket_name=settings.s3_bucket_name,
                prefix=settings.parquet_sink_prefix,
                tp=tp,
                state=state,
            )
            buffers.pop(tp, None)

    try:
        while True:
            timeout = max(0.1, float(settings.parquet_sink_flush_seconds))
            try:
                message = await asyncio.wait_for(consumer.getone(), timeout=timeout)
                tp = TopicPartition(message.topic, message.partition)
                state = buffers[tp]
                row = _normalize_message(message)
                state.rows.append(row)
                state.next_offset = message.offset + 1
                state.first_created_at = state.first_created_at or row.get("created_at")
                if not state.first_seen_monotonic:
                    state.first_seen_monotonic = loop.time()
            except asyncio.TimeoutError:
                pass

            await flush_due()
    finally:
        try:
            await flush_due(force_all=True)
        finally:
            await consumer.stop()


if __name__ == "__main__":
    asyncio.run(consume_to_parquet())
