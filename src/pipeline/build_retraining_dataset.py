from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.core.settings import get_settings
from src.app.services.drift import build_vocabulary
from src.app.services.event_archive import EventArchiveQuery, EventArchiveStore
from src.feature_store.feature_sets import AVAILABLE_DATA_COLUMNS, BASE_LABEL_COLUMN
from src.transformations.text import extract_text_features, preprocess_text
from src.utils import read_params


RELEVANT_EVENT_TYPES = {
    "moderation_report_routed_to_manual",
    "moderation_report_routed_to_ml",
    "moderation_decision_created",
    "comment_deleted",
}


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _parse_mapping(value: Any) -> dict[str, Any]:
    parsed = _maybe_json(value)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _extract_payload(row: pd.Series, prefix: str, suffix: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        clean_key = key.removeprefix(prefix).removesuffix(suffix)
        payload[clean_key] = _maybe_json(value)
    return payload


def _is_deleted_comment_text(text: str) -> bool:
    normalized = " ".join(str(text).strip().lower().split())
    return normalized in {
        "комментарий удален",
        "комментарий скрыт модератором",
    }


def build_retraining_frame(events: pd.DataFrame) -> pd.DataFrame:
    route_events = events[events["event_type"].isin({"moderation_report_routed_to_manual", "moderation_report_routed_to_ml"})].copy()
    decision_events = events[events["event_type"] == "moderation_decision_created"].copy()
    deleted_comment_ids = set(
        str(comment_id)
        for comment_id in events.loc[events["event_type"] == "comment_deleted", "aggregate_id"].dropna().astype(str).tolist()
    )

    if route_events.empty or decision_events.empty:
        return pd.DataFrame()

    route_events = route_events.sort_values("created_at").drop_duplicates(subset=["aggregate_id"], keep="last")
    decision_events = decision_events.sort_values("created_at").drop_duplicates(subset=["aggregate_id"], keep="last")

    merged = route_events.merge(
        decision_events,
        on="aggregate_id",
        suffixes=("_route", "_decision"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        route_payload = _extract_payload(row, "payload_", "_route")
        decision_payload = _extract_payload(row, "payload_", "_decision")
        comment_body = str(route_payload.get("comment_body") or "")
        if str(row.get("aggregate_id")) in deleted_comment_ids:
            continue
        if not comment_body or _is_deleted_comment_text(comment_body):
            continue
        text_prepared = str(route_payload.get("text_prepared") or preprocess_text(comment_body))
        text_features = _parse_mapping(route_payload.get("text_features"))
        if not text_features:
            text_features = extract_text_features(comment_body)
        user_features = _parse_mapping(route_payload.get("user_features"))
        verdict = str(decision_payload.get("verdict") or route_payload.get("verdict") or "")
        label = 1 if verdict == "toxic" else 0

        base_row: dict[str, Any] = {
            "report_id": row.get("aggregate_id"),
            "comment_id": route_payload.get("comment_id"),
            "comment_author_id": route_payload.get("comment_author_id"),
            "comment_body": comment_body,
            "tweet_text": comment_body,
            "cyberbullying_type": route_payload.get("reason") or "moderation_report",
            BASE_LABEL_COLUMN: label,
            "text_prepared": text_prepared,
            "decision_source": decision_payload.get("decision_source") or route_payload.get("decision_source"),
            "moderation_verdict": verdict,
            "ml_verdict": decision_payload.get("ml_verdict") or route_payload.get("ml_verdict"),
            "ml_score": decision_payload.get("confidence") or route_payload.get("confidence"),
            "ml_model_version": decision_payload.get("model_version") or route_payload.get("model_version"),
            "created_at": route_payload.get("created_at") or row.get("created_at_route"),
            "decision_created_at": decision_payload.get("created_at") or row.get("created_at_decision"),
        }
        base_row.update({key: value for key, value in text_features.items()})
        base_row.update({key: value for key, value in user_features.items()})
        rows.append(base_row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    for column in AVAILABLE_DATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0

    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    frame["decision_created_at"] = pd.to_datetime(frame["decision_created_at"], errors="coerce", utc=True)
    frame = frame.sort_values("created_at", ascending=True)
    return frame


def main() -> None:
    params = read_params()
    settings = get_settings()
    data_dir = Path(params.get("data", {}).get("output_dir", "data"))
    output_dir = data_dir / "retraining"
    output_dir.mkdir(parents=True, exist_ok=True)

    archive = EventArchiveStore(settings)
    window_hours = int(params.get("retraining", {}).get("window_hours", settings.retraining_data_window_hours))
    query = EventArchiveQuery(
        bucket_name=settings.s3_bucket_name,
        prefix=settings.parquet_sink_prefix,
        event_types=sorted(RELEVANT_EVENT_TYPES),
        days_back=max(1, window_hours // 24 or 1),
    )
    events = asyncio.run(archive.read_events(query))
    dataset = build_retraining_frame(events)
    output_path = output_dir / "retraining_dataset.parquet"
    dataset.to_parquet(output_path, index=False)

    vocab_path = output_dir / "baseline_vocab.json"
    if dataset.empty or "text_prepared" not in dataset.columns:
        vocab = set()
    else:
        vocab = build_vocabulary(dataset["text_prepared"].fillna("").astype(str).tolist())
    vocab_path.write_text(json.dumps(sorted(vocab), ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
