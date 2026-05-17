from __future__ import annotations

import asyncio
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.app.core.settings import get_settings
from src.feature_store.config import load_runtime_feature_config
from src.feature_store.feature_sets import BASE_LABEL_COLUMN
from src.utils import read_params


def _feature_profile(frame: pd.DataFrame, columns: list[str], bins: int = 10) -> dict[str, dict[str, list[float]]]:
    profile: dict[str, dict[str, list[float]]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(series.quantile(quantiles).to_numpy())
        if len(edges) < 2:
            continue
        histogram_edges = edges.copy()
        histogram_edges[0] = float("-inf")
        histogram_edges[-1] = float("inf")
        counts, _ = np.histogram(series, bins=histogram_edges)
        total = max(int(counts.sum()), 1)
        profile[column] = {
            "bin_edges": [float(edge) for edge in edges.tolist()],
            "bin_distribution": [float(count / total) for count in counts.tolist()],
        }
    return profile


def build_model(train_frame: pd.DataFrame, *, text_column: str, numeric_columns: list[str]) -> Pipeline:
    missing_columns = [column for column in [text_column, *numeric_columns] if column not in train_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing training columns: {missing_columns}")

    transformers: list[tuple[str, object, list[str] | str]] = [
        (
            "text",
            TfidfVectorizer(ngram_range=(1, 2), max_features=50000),
            text_column,
        ),
    ]
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("features", preprocessor), ("clf", clf)])


def main() -> None:
    params = read_params()
    model_params = params.get("model", {})
    settings = get_settings()
    data_dir = Path(params.get("data", {}).get("output_dir", "data"))
    train_frame = pd.read_parquet(data_dir / "train.parquet")

    feature_config = asyncio.run(
        load_runtime_feature_config(
            redis_url=settings.redis_url,
            namespace=settings.feature_store_namespace,
            params=params,
        )
    )

    pipeline = build_model(
        train_frame,
        text_column=feature_config.text_column,
        numeric_columns=list(feature_config.training_feature_columns),
    )
    pipeline.set_params(
        features__text__max_features=int(model_params.get("max_features", 50000)),
        features__text__ngram_range=(
            int(model_params.get("ngram_min", 1)),
            int(model_params.get("ngram_max", 2)),
        ),
        clf__max_iter=int(model_params.get("max_iter", 1000)),
        clf__class_weight=model_params.get("class_weight", "balanced"),
    )

    pipeline.fit(train_frame, train_frame[BASE_LABEL_COLUMN])

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cyberbullying_moderation.joblib"
    joblib.dump(pipeline, model_path)

    feature_profile_columns = list(feature_config.training_feature_columns)
    feature_profiles = _feature_profile(train_frame, feature_profile_columns)
    metadata_path = model_dir / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "numeric_features": feature_config.training_feature_columns,
                "text_column": feature_config.text_column,
                "feature_config_version": feature_config.version,
                "rows": len(train_frame),
                "feature_profiles": feature_profiles,
                "label_column": BASE_LABEL_COLUMN,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(model_path)


if __name__ == "__main__":
    main()
