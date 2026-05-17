from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.app.core.settings import get_settings
from src.feature_store.config import load_runtime_feature_config
from src.feature_store.feature_sets import BASE_LABEL_COLUMN
from src.pipeline.evaluate_model import score_split
from src.pipeline.train_model import _feature_profile, build_model
from src.utils import read_params


def _sample_fraction(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    fraction = max(min(fraction, 1.0), 0.0)
    if fraction >= 1.0:
        return frame.copy()
    if fraction <= 0.0:
        raise ValueError("Sample fraction must be greater than 0.")

    try:
        sample, _ = train_test_split(
            frame,
            train_size=fraction,
            random_state=seed,
            stratify=frame[BASE_LABEL_COLUMN],
        )
        return sample.copy()
    except Exception:
        return frame.sample(frac=fraction, random_state=seed).copy()


def _inject_label_noise(frame: pd.DataFrame, noise_rate: float, seed: int) -> tuple[pd.DataFrame, int]:
    noise_rate = max(min(noise_rate, 1.0), 0.0)
    if frame.empty or noise_rate <= 0.0:
        return frame.copy(), 0

    noisy = frame.copy()
    label_series = noisy[BASE_LABEL_COLUMN]
    if not pd.api.types.is_integer_dtype(label_series) and not pd.api.types.is_bool_dtype(label_series):
        label_series = label_series.astype(int)

    rng = np.random.default_rng(seed)
    noisy_count = max(1, int(round(len(noisy) * noise_rate)))
    noisy_count = min(noisy_count, len(noisy))
    noisy_indices = rng.choice(noisy.index.to_numpy(), size=noisy_count, replace=False)
    noisy.loc[noisy_indices, BASE_LABEL_COLUMN] = 1 - noisy.loc[noisy_indices, BASE_LABEL_COLUMN].astype(int)
    return noisy, int(noisy_count)


def _build_canary_model(frame: pd.DataFrame, *, text_column: str, numeric_columns: list[str], params: dict[str, object]):
    pipeline = build_model(frame, text_column=text_column, numeric_columns=numeric_columns)
    pipeline.set_params(
        features__text__max_features=int(params.get("max_features", 2000)),
        features__text__ngram_range=(
            int(params.get("ngram_min", 1)),
            int(params.get("ngram_max", 1)),
        ),
        clf__max_iter=int(params.get("max_iter", 400)),
        clf__class_weight=params.get("class_weight", "balanced"),
    )
    return pipeline


async def main() -> None:
    params = read_params()
    settings = get_settings()
    data_dir = Path(params.get("data", {}).get("output_dir", "data"))
    canary_params = params.get("canary_training", {})
    seed = int(params.get("split", {}).get("seed", 42))
    sample_fraction = float(canary_params.get("sample_fraction", 0.1))
    label_noise_rate = float(canary_params.get("label_noise_rate", 0.25))
    output_dir = Path(canary_params.get("output_dir", "models/canary_test"))
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_parquet(data_dir / "train.parquet")
    val_frame = pd.read_parquet(data_dir / "val.parquet")
    test_frame = pd.read_parquet(data_dir / "test.parquet")

    feature_config = await load_runtime_feature_config(
        redis_url=settings.redis_url,
        namespace=settings.feature_store_namespace,
        params=params,
    )

    sampled_train = _sample_fraction(train_frame, sample_fraction, seed)
    noisy_train, noisy_rows = _inject_label_noise(sampled_train, label_noise_rate, seed)

    model = _build_canary_model(
        noisy_train,
        text_column=feature_config.text_column,
        numeric_columns=list(feature_config.training_feature_columns),
        params=canary_params,
    )
    model.fit(noisy_train, noisy_train[BASE_LABEL_COLUMN])

    model_path = output_dir / "cyberbullying_moderation.joblib"
    joblib.dump(model, model_path)

    feature_profiles = _feature_profile(noisy_train, list(feature_config.training_feature_columns))
    metadata = {
        "numeric_features": list(feature_config.training_feature_columns),
        "text_column": feature_config.text_column,
        "feature_config_version": feature_config.version,
        "rows": len(noisy_train),
        "sample_fraction": sample_fraction,
        "label_noise_rate": label_noise_rate,
        "noisy_rows": noisy_rows,
        "base_train_rows": len(train_frame),
        "feature_profiles": feature_profiles,
        "label_column": BASE_LABEL_COLUMN,
        "source": "canary_test_pipeline",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    report = {
        "model_name": settings.model_registry_model_name,
        "mode": "canary_test",
        "sample_fraction": sample_fraction,
        "label_noise_rate": label_noise_rate,
        "train_rows": len(noisy_train),
        "base_train_rows": len(train_frame),
        "noisy_rows": noisy_rows,
        "validation": score_split(model, val_frame),
        "test": score_split(model, test_frame),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = reports_dir / "canary_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(model_path)


if __name__ == "__main__":
    asyncio.run(main())
