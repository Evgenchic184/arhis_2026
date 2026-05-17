from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import select

from src.app.core.database import async_session_maker
from src.app.core.settings import get_settings
from src.app.models.comment import Comment
from src.app.models.enums import CommentVisibility
from src.app.services.model_registry import ModelRegistryService
from src.app.services.retraining_datasets import RetrainingDatasetArchiveService, RetrainingDatasetBundle
from src.feature_store.config import load_runtime_feature_config
from src.feature_store.feature_sets import BASE_LABEL_COLUMN
from src.pipeline.build_retraining_dataset import build_retraining_frame
from src.pipeline.evaluate_model import score_split
from src.pipeline.train_model import _feature_profile, build_model
from src.utils import read_params


def _parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        if safe[column].dtype == "object":
            safe[column] = safe[column].map(
                lambda value: None if pd.isna(value) else str(value)
            )
    return safe


def _split_dataset(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Retraining dataset is empty.")
    train_frame, holdout_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=seed,
        stratify=frame[BASE_LABEL_COLUMN],
    )
    val_frame, test_frame = train_test_split(
        holdout_frame,
        test_size=0.5,
        random_state=seed,
        stratify=holdout_frame[BASE_LABEL_COLUMN],
    )
    return train_frame, val_frame, test_frame


def _describe_dataset(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {"rows": int(len(frame)), "columns": list(frame.columns)}
    if frame.empty or BASE_LABEL_COLUMN not in frame.columns:
        return summary
    label_counts = frame[BASE_LABEL_COLUMN].value_counts(dropna=False).to_dict()
    summary["label_counts"] = {str(key): int(value) for key, value in label_counts.items()}
    summary["label_positive_rate"] = float(frame[BASE_LABEL_COLUMN].mean())
    return summary


def _is_splittable(frame: pd.DataFrame, min_class_count: int) -> tuple[bool, str]:
    if frame.empty:
        return False, "Retraining dataset is empty."
    if BASE_LABEL_COLUMN not in frame.columns:
        return False, f"Missing required label column: {BASE_LABEL_COLUMN}."
    label_counts = frame[BASE_LABEL_COLUMN].value_counts(dropna=False)
    if label_counts.empty:
        return False, "Retraining dataset has no labels."
    observed_min_class_count = int(label_counts.min())
    if observed_min_class_count < min_class_count:
        return (
            False,
            f"Retraining dataset is too small for stratified split: minimum class count is {observed_min_class_count}, need at least {min_class_count} per class.",
        )
    if len(frame) < min_class_count * 2:
        return (
            False,
            f"Retraining dataset is too small for train/val/test split: {len(frame)} rows, need at least {min_class_count * 2}.",
        )
    return True, ""


def _select_training_fallback(
    system_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    *,
    target_min_class_count: int,
    max_train_share: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if system_frame.empty or train_frame.empty or BASE_LABEL_COLUMN not in system_frame.columns or BASE_LABEL_COLUMN not in train_frame.columns:
        return pd.DataFrame(), {
            "system_rows": int(len(system_frame)),
            "train_rows_used": 0,
            "target_min_class_count": target_min_class_count,
            "max_train_share": max_train_share,
            "status": "insufficient_system_or_train_data",
        }

    system_counts = system_frame[BASE_LABEL_COLUMN].value_counts(dropna=False)
    train_counts = train_frame[BASE_LABEL_COLUMN].value_counts(dropna=False)
    if system_counts.empty:
        return pd.DataFrame(), {
            "system_rows": int(len(system_frame)),
            "train_rows_used": 0,
            "target_min_class_count": target_min_class_count,
            "max_train_share": max_train_share,
            "status": "no_system_labels",
        }

    system_rows = int(len(system_frame))
    if system_rows <= 0:
        return pd.DataFrame(), {
            "system_rows": 0,
            "train_rows_used": 0,
            "target_min_class_count": target_min_class_count,
            "max_train_share": max_train_share,
            "status": "no_system_rows",
        }

    max_train_rows = int(system_rows * max_train_share / max(1.0 - max_train_share, 1e-9))
    if max_train_rows <= 0:
        return pd.DataFrame(), {
            "system_rows": system_rows,
            "train_rows_used": 0,
            "target_min_class_count": target_min_class_count,
            "max_train_share": max_train_share,
            "status": "train_cap_exhausted",
        }

    supplemental_parts: list[pd.DataFrame] = []
    rows_used = 0
    train_remaining = train_frame.copy()
    class_counts = {int(label): int(count) for label, count in system_counts.items()}
    all_labels = sorted({int(label) for label in set(system_counts.index.tolist()) | set(train_counts.index.tolist())})

    deficits = {
        label: max(int(target_min_class_count) - count, 0)
        for label in all_labels
        for count in [class_counts.get(label, 0)]
    }
    for label, deficit in sorted(deficits.items(), key=lambda item: item[1], reverse=True):
        if deficit <= 0 or rows_used >= max_train_rows:
            continue
        candidates = train_remaining[train_remaining[BASE_LABEL_COLUMN] == label]
        if candidates.empty:
            continue
        sample_n = min(deficit, len(candidates), max_train_rows - rows_used)
        if sample_n <= 0:
            continue
        sampled = candidates.sample(n=sample_n, random_state=seed + int(label) + rows_used).copy()
        sampled["dataset_origin"] = "train_fallback"
        supplemental_parts.append(sampled)
        rows_used += len(sampled)
        train_remaining = train_remaining.drop(sampled.index, errors="ignore")
        class_counts[label] = class_counts.get(label, 0) + len(sampled)

    supplemental_frame = pd.concat(supplemental_parts, ignore_index=True) if supplemental_parts else pd.DataFrame()
    return supplemental_frame, {
        "system_rows": system_rows,
        "train_rows_used": rows_used,
        "target_min_class_count": target_min_class_count,
        "max_train_share": max_train_share,
        "status": "ready" if rows_used > 0 else "system_only",
        "class_counts_after_fallback": {str(label): int(count) for label, count in class_counts.items()},
    }


async def _filter_deleted_comments(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty or "comment_id" not in frame.columns:
        return frame, 0

    comment_ids = [
        str(comment_id)
        for comment_id in frame["comment_id"].dropna().astype(str).tolist()
        if str(comment_id)
    ]
    if not comment_ids:
        return frame, 0

    async with async_session_maker() as db:
        result = await db.execute(
            select(Comment.id).where(Comment.id.in_(comment_ids)).where(Comment.visibility != CommentVisibility.deleted)
        )
        allowed_ids = {str(comment_id) for comment_id in result.scalars().all()}

    filtered = frame[frame["comment_id"].astype(str).isin(allowed_ids)].copy()
    excluded = int(len(frame) - len(filtered))
    return filtered, excluded


async def main() -> None:
    params = read_params()
    settings = get_settings()
    seed = int(params.get("split", {}).get("seed", 42))
    data_dir = Path(params.get("data", {}).get("output_dir", "data"))
    retraining_dir = data_dir / "retraining"
    retraining_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports/retraining")
    reports_dir.mkdir(parents=True, exist_ok=True)
    retraining_params = params.get("retraining", {})
    target_min_class_count = int(retraining_params.get("min_class_count", 100))
    max_train_share = float(retraining_params.get("fallback_train_max_share", 0.8))

    archive_path = retraining_dir / "retraining_dataset.parquet"
    train_fallback_path = data_dir / "train.parquet"
    from src.app.services.event_archive import EventArchiveQuery, EventArchiveStore

    archive = EventArchiveStore(settings)
    query = EventArchiveQuery(
        bucket_name=settings.s3_bucket_name,
        prefix=settings.parquet_sink_prefix,
        event_types=[
            "moderation_report_routed_to_manual",
            "moderation_report_routed_to_ml",
            "moderation_decision_created",
            "comment_deleted",
        ],
        days_back=max(1, settings.retraining_data_window_hours // 24 or 1),
    )
    events = await archive.read_events(query)
    frame = build_retraining_frame(events)
    source = "event_archive"
    if frame.empty and archive_path.exists():
        frame = pd.read_parquet(archive_path)
        source = "cached_parquet"
    elif not frame.empty:
        frame.to_parquet(archive_path, index=False)

    if frame.empty:
        raise ValueError("Retraining dataset is empty.")

    frame, excluded_deleted_comments = await _filter_deleted_comments(frame)
    system_frame = frame.copy()
    supplemental_frame = pd.DataFrame()
    fallback_summary: dict[str, object] = {
        "system_rows": int(len(system_frame)),
        "train_rows_used": 0,
        "target_min_class_count": target_min_class_count,
        "max_train_share": max_train_share,
        "status": "not_attempted",
    }

    if len(system_frame) > 0:
        if train_fallback_path.exists():
            train_fallback_frame = pd.read_parquet(train_fallback_path)
            supplemental_frame, fallback_summary = _select_training_fallback(
                system_frame,
                train_fallback_frame,
                target_min_class_count=target_min_class_count,
                max_train_share=max_train_share,
                seed=seed,
            )
        else:
            fallback_summary = {
                "system_rows": int(len(system_frame)),
                "train_rows_used": 0,
                "target_min_class_count": target_min_class_count,
                "max_train_share": max_train_share,
                "status": "train_fallback_missing",
            }

    if not supplemental_frame.empty:
        frame = pd.concat([system_frame, supplemental_frame], ignore_index=True, sort=False)
    else:
        frame = system_frame

    _parquet_safe_frame(frame).to_parquet(archive_path, index=False)
    dataset_summary = _describe_dataset(frame)
    dataset_summary["source"] = source if supplemental_frame.empty else f"{source}+train_fallback"
    dataset_summary["excluded_deleted_comments"] = excluded_deleted_comments
    dataset_summary["system_rows"] = int(len(system_frame))
    dataset_summary["supplemental_rows"] = int(len(supplemental_frame))
    dataset_summary["train_share"] = (
        float(len(supplemental_frame) / len(frame)) if len(frame) > 0 else 0.0
    )
    dataset_summary["fallback"] = fallback_summary
    summary_path = reports_dir / "dataset_summary.json"
    feature_config = await load_runtime_feature_config(
        redis_url=settings.redis_url,
        namespace=settings.feature_store_namespace,
        params=params,
    )
    version = f"{settings.model_registry_model_name}-{feature_config.version}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    dataset_summary_path = str(summary_path)
    dataset_summary["version"] = version
    can_split, split_reason = _is_splittable(frame, target_min_class_count)
    dataset_summary["split_status"] = "ready" if can_split else "skipped_insufficient_data"
    if not can_split:
        dataset_summary["split_reason"] = split_reason
    summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = reports_dir / "dataset_manifest.json"
    manifest = {
        "model_name": settings.model_registry_model_name,
        "version": version,
        "feature_config_version": feature_config.version,
        "source": dataset_summary["source"],
        "dataset_rows": int(len(frame)),
        "label_counts": dataset_summary.get("label_counts", {}),
        "label_positive_rate": dataset_summary.get("label_positive_rate"),
        "split_status": dataset_summary["split_status"],
        "split_reason": dataset_summary.get("split_reason"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    archive_service = RetrainingDatasetArchiveService(settings)
    snapshot_paths = await archive_service.upload_snapshot(
        version=version,
        dataset_path=archive_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )

    if not can_split:
        validation_report = {
            "status": "skipped_insufficient_data",
            "reason": split_reason,
            "dataset_summary_path": dataset_summary_path,
            "dataset_archive": snapshot_paths,
            "thresholds": {
                "required_accuracy": settings.model_registry_required_accuracy,
                "canary_traffic_percent": settings.model_registry_canary_traffic_percent,
            },
        }
        (reports_dir / "validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")
        (reports_dir / "model_registry_report.json").write_text(
            json.dumps(
                {
                    "model_name": settings.model_registry_model_name,
                    "version": version,
                    "status": "skipped_insufficient_data",
                    "reason": split_reason,
                    "feature_config_version": feature_config.version,
                    "dataset_rows": int(len(frame)),
                    "label_counts": dataset_summary.get("label_counts", {}),
                    "dataset_archive": snapshot_paths,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(summary_path)
        return

    try:
        train_frame, val_frame, test_frame = _split_dataset(frame, seed=seed)
    except ValueError as exc:
        split_reason = str(exc)
        dataset_summary["split_status"] = "skipped_insufficient_data"
        dataset_summary["split_reason"] = split_reason
        summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        validation_report = {
            "status": "skipped_insufficient_data",
            "reason": split_reason,
            "dataset_summary_path": dataset_summary_path,
            "dataset_archive": snapshot_paths,
            "thresholds": {
                "required_accuracy": settings.model_registry_required_accuracy,
                "canary_traffic_percent": settings.model_registry_canary_traffic_percent,
            },
        }
        (reports_dir / "validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")
        (reports_dir / "model_registry_report.json").write_text(
            json.dumps(
                {
                    "model_name": settings.model_registry_model_name,
                    "version": version,
                    "status": "skipped_insufficient_data",
                    "reason": split_reason,
                    "feature_config_version": feature_config.version,
                    "dataset_rows": int(len(frame)),
                    "label_counts": dataset_summary.get("label_counts", {}),
                    "dataset_archive": snapshot_paths,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(summary_path)
        return
    train_path = retraining_dir / "train.parquet"
    val_path = retraining_dir / "val.parquet"
    test_path = retraining_dir / "test.parquet"
    _parquet_safe_frame(train_frame).to_parquet(train_path, index=False)
    _parquet_safe_frame(val_frame).to_parquet(val_path, index=False)
    _parquet_safe_frame(test_frame).to_parquet(test_path, index=False)

    candidate = build_model(
        train_frame,
        text_column=feature_config.text_column,
        numeric_columns=list(feature_config.training_feature_columns),
    )
    candidate.set_params(features__text__max_features=50000, features__text__ngram_range=(1, 2), clf__max_iter=1000)
    candidate.fit(train_frame, train_frame[BASE_LABEL_COLUMN])

    model_path = Path("models/retraining")
    model_path.mkdir(parents=True, exist_ok=True)
    artifact_path = model_path / "cyberbullying_moderation.joblib"
    joblib.dump(candidate, artifact_path)

    feature_profiles = _feature_profile(train_frame, list(feature_config.training_feature_columns))
    metadata_path = model_path / "training_metadata.json"
    metadata = {
        "numeric_features": list(feature_config.training_feature_columns),
        "text_column": feature_config.text_column,
        "feature_config_version": feature_config.version,
        "rows": len(train_frame),
        "feature_profiles": feature_profiles,
        "label_column": BASE_LABEL_COLUMN,
        "source": "retraining_pipeline",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    validation_report = {
        "validation": score_split(candidate, val_frame),
        "test": score_split(candidate, test_frame),
        "thresholds": {
            "required_accuracy": settings.model_registry_required_accuracy,
            "canary_traffic_percent": settings.model_registry_canary_traffic_percent,
        },
        "dataset_summary_path": dataset_summary_path,
    }
    report_path = reports_dir / "validation_report.json"
    report_path.write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")
    registry_report_path = reports_dir / "model_registry_report.json"

    baseline_vocab_path = retraining_dir / "baseline_vocab.json"
    if not baseline_vocab_path.exists():
        baseline_vocab_path.write_text(
            json.dumps(
                sorted(
                    {
                        token
                        for text in train_frame.get("text_prepared", pd.Series(dtype=str)).fillna("").astype(str).tolist()
                        for token in str(text).lower().split()
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest.update(
        {
            "train_rows": int(len(train_frame)),
            "val_rows": int(len(val_frame)),
            "test_rows": int(len(test_frame)),
            "split_status": "ready",
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    archive_paths = await archive_service.upload_bundle(
        RetrainingDatasetBundle(
            version=version,
            dataset_path=archive_path,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            baseline_vocab_path=baseline_vocab_path,
            summary_path=summary_path,
            manifest_path=manifest_path,
        )
    )
    validation_report["dataset_archive"] = archive_paths
    report_path.write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    registry = ModelRegistryService(settings)
    passed, sample_metrics, _ = await registry.validate_candidate(
        candidate,
        val_frame,
        sample_size=settings.model_registry_validation_sample_size,
        required_accuracy=settings.model_registry_required_accuracy,
    )

    async with async_session_maker() as db:
        if not passed:
            await registry.mark_rejected(
                db,
                model_name=settings.model_registry_model_name,
                version=version,
                local_model_path=artifact_path,
                metadata_path=metadata_path,
                validation_report_path=report_path,
                feature_config_version=feature_config.version,
                validation_accuracy=sample_metrics["accuracy"],
                validation_sample_size=settings.model_registry_validation_sample_size,
                validation_metrics=sample_metrics,
                training_metadata=metadata,
                notes="Validation gate failed on retraining sample.",
            )
            await db.commit()
            registry_report_path.write_text(
                json.dumps(
                    {
                        "model_name": settings.model_registry_model_name,
                        "version": version,
                        "status": "rejected",
                        "validation_metrics": sample_metrics,
                        "feature_config_version": feature_config.version,
                        "dataset_archive": archive_paths,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise SystemExit(1)

        row = await registry.register_version(
            db,
            model_name=settings.model_registry_model_name,
            version=version,
            local_model_path=artifact_path,
            metadata_path=metadata_path,
            validation_report_path=report_path,
            feature_config_version=feature_config.version,
            validation_accuracy=sample_metrics["accuracy"],
            validation_sample_size=settings.model_registry_validation_sample_size,
            validation_metrics=sample_metrics,
            training_metadata=metadata,
            canary_traffic_percent=settings.model_registry_canary_traffic_percent,
        )
        await db.commit()

    registry_report_path.write_text(
        json.dumps(
            {
                "model_name": row.model_name,
                "version": row.version,
                "status": row.status.value,
                "artifact_uri": row.artifact_uri,
                "metadata_uri": row.metadata_uri,
                "traffic_percent": row.traffic_percent,
                "dataset_archive": archive_paths,
                "validation_metrics": sample_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_name": row.model_name,
                "version": row.version,
                "status": row.status.value,
                "artifact_uri": row.artifact_uri,
                "dataset_archive": archive_paths,
                "validation_metrics": sample_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
