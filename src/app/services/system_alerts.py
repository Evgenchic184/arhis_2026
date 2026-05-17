from __future__ import annotations

import asyncio
import hashlib
import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.enums import AlertStatus
from src.app.models.system_alert import SystemAlert
from src.app.core.settings import Settings, get_settings


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _build_live_fingerprint(labels: dict[str, Any], starts_at: Any, receiver: str | None) -> str:
    payload = json.dumps(
        {
            "labels": labels,
            "starts_at": starts_at,
            "receiver": receiver,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fetch_prometheus_alerts_sync(base_url: str) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/v1/alerts"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "success":
        return []
    data = payload.get("data") or {}
    alerts = data.get("alerts") or []
    return alerts if isinstance(alerts, list) else []


class SystemAlertService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def ingest_webhook(self, db: AsyncSession, payload: dict[str, Any]) -> int:
        receiver = payload.get("receiver")
        alerts = payload.get("alerts") or []
        now = datetime.now(timezone.utc)
        ingested = 0

        for item in alerts:
            labels = _json_safe_value(item.get("labels") or {})
            annotations = _json_safe_value(item.get("annotations") or {})
            fingerprint = str(item.get("fingerprint") or "")
            if not fingerprint:
                continue

            alertname = str(labels.get("alertname") or "unknown")
            severity = labels.get("severity")
            status_raw = str(item.get("status") or payload.get("status") or "firing").lower()
            status = AlertStatus.firing if status_raw != AlertStatus.resolved.value else AlertStatus.resolved
            summary = annotations.get("summary")
            description = annotations.get("description")
            starts_at = _parse_datetime(item.get("startsAt"))
            ends_at = _parse_datetime(item.get("endsAt"))
            resolved_at = now if status == AlertStatus.resolved else None
            is_active = status == AlertStatus.firing

            row_data = {
                "fingerprint": fingerprint,
                "status": status.value,
                "alertname": alertname,
                "severity": severity,
                "summary": summary,
                "description": description,
                "labels": labels,
                "annotations": annotations,
                "raw_payload": _json_safe_value(payload),
                "receiver": receiver,
                "generator_url": item.get("generatorURL"),
                "starts_at": starts_at,
                "ends_at": ends_at,
                "resolved_at": resolved_at,
                "received_at": now,
                "is_active": is_active,
            }

            stmt = insert(SystemAlert).values(row_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=[SystemAlert.fingerprint],
                set_={
                    "status": row_data["status"],
                    "alertname": row_data["alertname"],
                    "severity": row_data["severity"],
                    "summary": row_data["summary"],
                    "description": row_data["description"],
                    "labels": row_data["labels"],
                    "annotations": row_data["annotations"],
                    "raw_payload": row_data["raw_payload"],
                    "receiver": row_data["receiver"],
                    "generator_url": row_data["generator_url"],
                    "starts_at": row_data["starts_at"],
                    "ends_at": row_data["ends_at"],
                    "resolved_at": row_data["resolved_at"],
                    "received_at": row_data["received_at"],
                    "is_active": row_data["is_active"],
                },
            )
            await db.execute(stmt)
            ingested += 1

        return ingested

    async def _load_live_prometheus_alerts(self) -> list[SystemAlert]:
        try:
            alerts = await asyncio.to_thread(_fetch_prometheus_alerts_sync, self.settings.prometheus_base_url)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return []

        now = datetime.now(timezone.utc)
        live_rows: list[SystemAlert] = []
        for item in alerts:
            labels = _json_safe_value(item.get("labels") or {})
            annotations = _json_safe_value(item.get("annotations") or {})
            state = str(item.get("state") or "firing").lower()
            if state not in {"firing", "pending"}:
                continue

            starts_at = _parse_datetime(item.get("activeAt") or item.get("startsAt"))
            fingerprint = str(item.get("fingerprint") or "") or _build_live_fingerprint(
                labels,
                starts_at.isoformat() if starts_at else None,
                "prometheus",
            )
            alertname = str(labels.get("alertname") or item.get("labels", {}).get("alertname") or "unknown")
            severity = labels.get("severity")
            summary = annotations.get("summary")
            description = annotations.get("description")

            live_rows.append(
                SystemAlert(
                    fingerprint=fingerprint,
                    status=AlertStatus.firing,
                    alertname=alertname,
                    severity=severity,
                    summary=summary,
                    description=description,
                    labels=labels,
                    annotations=annotations,
                    raw_payload=_json_safe_value({"source": "prometheus", "alert": item}),
                    receiver="prometheus",
                    generator_url=item.get("generatorURL"),
                    starts_at=starts_at,
                    ends_at=None,
                    resolved_at=None,
                    received_at=now,
                    is_active=True,
                )
            )
        return live_rows

    async def list_alerts(
        self,
        db: AsyncSession,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SystemAlert]:
        stmt = select(SystemAlert).order_by(
            SystemAlert.is_active.desc(),
            desc(SystemAlert.updated_at),
        )
        if status and status != "all":
            if status == "active":
                stmt = stmt.where(SystemAlert.is_active.is_(True))
            elif status == "resolved":
                stmt = stmt.where(SystemAlert.is_active.is_(False))
        stmt = stmt.limit(max(1, min(limit, 500)))
        result = await db.execute(stmt)
        rows = list(result.scalars().all())

        live_alerts = await self._load_live_prometheus_alerts()
        existing_fingerprints = {row.fingerprint for row in rows}
        for alert in live_alerts:
            if alert.fingerprint in existing_fingerprints:
                continue
            rows.append(alert)
            existing_fingerprints.add(alert.fingerprint)

        rows.sort(key=lambda item: (item.is_active, item.updated_at), reverse=True)
        if status == "active":
            rows = [row for row in rows if row.is_active]
        elif status == "resolved":
            rows = [row for row in rows if not row.is_active]

        return rows[: max(1, min(limit, 500))]


system_alert_service = SystemAlertService()
