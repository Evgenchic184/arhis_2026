from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.app.core.database import async_session_maker
from src.app.core.settings import get_settings
from src.app.services.model_registry import ModelRegistryService
from src.feature_store.feature_sets import BASE_LABEL_COLUMN
from src.utils import read_params


def _json_safe_value(value):
    if isinstance(value, (float,)) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    return value


def _build_version_name(model_name: str, feature_config_version: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{model_name}-canary-{feature_config_version}-{timestamp}"


async def main() -> None:
    params = read_params()
    settings = get_settings()
    data_dir = Path(params.get("data", {}).get("output_dir", "data"))
    canary_params = params.get("canary_training", {})
    model_name = settings.model_registry_model_name
    model_dir = Path(canary_params.get("output_dir", "models/canary_test"))
    model_path = model_dir / "cyberbullying_moderation.joblib"
    metadata_path = model_dir / "training_metadata.json"
    validation_report_path = Path("reports/canary_validation_report.json")
    validation_frame = pd.read_parquet(data_dir / "val.parquet")
    model = joblib.load(model_path)
    registry = ModelRegistryService(settings)

    feature_config_version = int(params.get("features", {}).get("runtime", {}).get("version", settings.feature_config_version))
    version = _build_version_name(model_name, feature_config_version)
    required_accuracy = float(canary_params.get("validation_required_accuracy", 0.0))
    sample_size = settings.model_registry_validation_sample_size
    passed, sample_metrics, sample_frame = await registry.validate_candidate(
        model,
        validation_frame,
        sample_size=sample_size,
        required_accuracy=required_accuracy,
    )

    training_metadata = None
    if metadata_path.exists():
        training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training_metadata = _json_safe_value(training_metadata)

    registry_report = {
        "model_name": model_name,
        "version": version,
        "feature_config_version": feature_config_version,
        "required_accuracy": required_accuracy,
        "validation_sample_size": len(sample_frame),
        "validation_metrics": sample_metrics,
        "status": "rejected" if not passed else "pending",
        "label_column": BASE_LABEL_COLUMN,
        "sample_rows": len(sample_frame),
        "mode": "canary_test",
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "canary_model_registry_report.json"

    async with async_session_maker() as db:
        if not passed:
            row = await registry.mark_rejected(
                db,
                model_name=model_name,
                version=version,
                local_model_path=model_path,
                metadata_path=metadata_path if metadata_path.exists() else None,
                validation_report_path=validation_report_path if validation_report_path.exists() else None,
                feature_config_version=feature_config_version,
                validation_accuracy=sample_metrics["accuracy"],
                validation_sample_size=len(sample_frame),
                validation_metrics=sample_metrics,
                training_metadata=training_metadata,
                notes="Canary validation gate failed.",
            )
            registry_report.update(
                {
                    "status": row.status.value,
                    "artifact_uri": row.artifact_uri,
                    "metadata_uri": row.metadata_uri,
                }
            )
            await db.commit()
            report_path.write_text(json.dumps(registry_report, ensure_ascii=False, indent=2), encoding="utf-8")
            raise SystemExit(1)

        row = await registry.register_version(
            db,
            model_name=model_name,
            version=version,
            local_model_path=model_path,
            metadata_path=metadata_path if metadata_path.exists() else None,
            validation_report_path=validation_report_path if validation_report_path.exists() else None,
            feature_config_version=feature_config_version,
            validation_accuracy=sample_metrics["accuracy"],
            validation_sample_size=len(sample_frame),
            validation_metrics=sample_metrics,
            training_metadata=training_metadata,
            canary_traffic_percent=settings.model_registry_canary_traffic_percent,
        )
        await db.commit()
        registry_report.update(
            {
                "status": row.status.value,
                "artifact_uri": row.artifact_uri,
                "metadata_uri": row.metadata_uri,
                "traffic_percent": row.traffic_percent,
            }
        )

    report_path.write_text(json.dumps(registry_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    asyncio.run(main())
