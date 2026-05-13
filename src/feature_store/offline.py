from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.feature_store.schemas import CommentFeatureSnapshot


def generate_user_ids(n_rows: int, n_users: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_users, size=n_rows)


def generate_synthetic_user_features(user_key: str, seed: int = 42) -> dict[str, Any]:
    stable_hash = hashlib.sha256(f"{user_key}:{seed}".encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(stable_hash[:16], 16) % (2**32))
    account_age_days = int(rng.exponential(scale=365))
    is_new_user = int(account_age_days < 7)
    base_reputation = float(rng.beta(2, 2))
    reputation_score = base_reputation * (0.7 if is_new_user else 1.0)
    reports_last_30d = int(rng.poisson(max(0.0, 3 - 2 * reputation_score)))
    reports_last_7d = int(max(0, round(reports_last_30d * 0.35)))
    reports_last_24h = int(max(0, round(reports_last_30d * 0.12)))
    deleted_comments_last_30d = int(rng.poisson(max(0.0, 2.5 - 1.5 * reputation_score)))
    deleted_comments_last_7d = int(max(0, round(deleted_comments_last_30d * 0.35)))
    deleted_comments_last_1d = int(max(0, round(deleted_comments_last_30d * 0.12)))
    hidden_comments_last_30d = int(rng.poisson(max(0.0, 1.5 - 0.8 * reputation_score)))
    hidden_comments_last_7d = int(max(0, round(hidden_comments_last_30d * 0.35)))
    hidden_comments_last_1d = int(max(0, round(hidden_comments_last_30d * 0.12)))
    comment_count_last_30d = int(rng.poisson(20 + 40 * reputation_score))
    comment_count_last_7d = int(max(0, round(comment_count_last_30d * 0.25)))
    comment_count_last_1d = int(max(0, round(comment_count_last_30d * 0.05)))
    auto_action_count_last_30d = int(round(comment_count_last_30d * float(np.clip(reputation_score, 0, 1))))
    manual_overrule_count_last_30d = int(round(comment_count_last_30d * float(np.clip(1 - reputation_score, 0, 1))))
    auto_action_rate_last_30d = auto_action_count_last_30d / max(comment_count_last_30d, 1)
    manual_overrule_rate_last_30d = manual_overrule_count_last_30d / max(comment_count_last_30d, 1)

    return {
        "user_id": user_key,
        "is_new_user": is_new_user,
        "account_age_days": account_age_days,
        "reputation_score": float(np.clip(reputation_score, 0, 1)),
        "reports_last_24h": reports_last_24h,
        "reports_last_7d": reports_last_7d,
        "reports_last_30d": reports_last_30d,
        "deleted_comments_last_1d": deleted_comments_last_1d,
        "deleted_comments_last_7d": deleted_comments_last_7d,
        "deleted_comments_last_30d": deleted_comments_last_30d,
        "hidden_comments_last_1d": hidden_comments_last_1d,
        "hidden_comments_last_7d": hidden_comments_last_7d,
        "hidden_comments_last_30d": hidden_comments_last_30d,
        "comment_count_last_1d": comment_count_last_1d,
        "comment_count_last_7d": comment_count_last_7d,
        "comment_count_last_30d": comment_count_last_30d,
        "auto_action_count_last_30d": auto_action_count_last_30d,
        "manual_overrule_count_last_30d": manual_overrule_count_last_30d,
        "auto_action_rate_last_30d": auto_action_rate_last_30d,
        "manual_overrule_rate_last_30d": manual_overrule_rate_last_30d,
        "last_ml_confidence": float(np.clip(reputation_score, 0, 1)),
        "last_ml_verdict": "unknown",
    }


class OfflineFeatureStore:
    def __init__(self, root_path: str | Path = "data/offline_feature_store", seed: int = 42) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def write_training_examples(
        self,
        examples: Iterable[CommentFeatureSnapshot],
        dataset_name: str = "training_examples",
    ) -> Path:
        rows = [example.to_flat_dict() for example in examples]
        if not rows:
            raise ValueError("No training examples were provided.")

        frame = pd.DataFrame(rows)
        output_path = self.root_path / f"{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.parquet"
        frame.to_parquet(output_path, index=False)
        return output_path

    def materialize_from_frame(
        self,
        frame: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        return output_path

    def read_all(self) -> pd.DataFrame:
        files = sorted(self.root_path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(file) for file in files]
        return pd.concat(frames, ignore_index=True)
