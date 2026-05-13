from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.feature_store.feature_sets import BASE_LABEL_COLUMN
from src.utils import read_params


def score_split(model, frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame[BASE_LABEL_COLUMN]
    y_pred = model.predict(frame)
    y_proba = model.predict_proba(frame)[:, 1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main() -> None:
    params = read_params()
    data_dir = Path(params.get("data", {}).get("output_dir", "data"))
    model_path = Path("models/cyberbullying_moderation.joblib")

    model = joblib.load(model_path)
    val_frame = pd.read_parquet(data_dir / "val.parquet")
    test_frame = pd.read_parquet(data_dir / "test.parquet")

    report = {
        "validation": score_split(model, val_frame),
        "test": score_split(model, test_frame),
        "thresholds": {
            "low": params.get("features", {}).get("confidence_threshold_low", 0.65),
            "high": params.get("features", {}).get("confidence_threshold_high", 0.9),
        },
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
