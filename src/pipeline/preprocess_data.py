from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_store.feature_sets import (
    AVAILABLE_DATA_COLUMNS,
    AVAILABLE_USER_FEATURE_COLUMNS,
    BASE_LABEL_COLUMN,
    BASE_IDENTIFIER_COLUMN,
)
from src.feature_store.offline import OfflineFeatureStore, generate_synthetic_user_features, generate_user_ids
from src.transformations.text import extract_text_features, preprocess_text
from src.utils import read_params


def build_dataset(raw_csv: Path, output_dir: Path, synthetic_users: int, seed: int) -> tuple[Path, Path, Path, Path, Path]:
    df = pd.read_csv(raw_csv)
    if "tweet_text" not in df.columns or "cyberbullying_type" not in df.columns:
        raise ValueError("Expected tweet_text and cyberbullying_type columns in the raw dataset.")

    df["cyberbullying_bin"] = (df["cyberbullying_type"] != "not_cyberbullying").astype(int)
    df["text_prepared"] = df["tweet_text"].map(preprocess_text)
    df["user_id"] = generate_user_ids(len(df), n_users=synthetic_users, seed=seed)

    text_feature_rows = [extract_text_features(text) for text in df["tweet_text"]]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(text_feature_rows)], axis=1)

    user_feature_rows = []
    for user_id in sorted(df["user_id"].unique()):
        user_feature_rows.append(generate_synthetic_user_features(str(user_id), seed=seed))
    user_frame = pd.DataFrame(user_feature_rows)[[BASE_IDENTIFIER_COLUMN] + AVAILABLE_USER_FEATURE_COLUMNS]
    df = df.merge(user_frame, on="user_id", how="left", suffixes=("", "_online"))

    train_frame, holdout_frame = train_test_split(
        df,
        test_size=float(read_params().get("split", {}).get("holdout_size", 0.2)),
        random_state=seed,
        stratify=df[BASE_LABEL_COLUMN],
    )

    val_share = float(read_params().get("split", {}).get("val_share_of_holdout", 0.5))
    val_frame, test_frame = train_test_split(
        holdout_frame,
        test_size=1 - val_share,
        random_state=seed,
        stratify=holdout_frame[BASE_LABEL_COLUMN],
    )

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    user_path = output_dir / "with_user_features.parquet"

    selected_columns = [column for column in AVAILABLE_DATA_COLUMNS if column in df.columns]
    feature_store = OfflineFeatureStore(output_dir / "offline_feature_store", seed=seed)
    feature_store.materialize_from_frame(df[selected_columns], output_dir / "all_data.parquet")

    train_frame[selected_columns].to_parquet(train_path, index=False)
    val_frame[selected_columns].to_parquet(val_path, index=False)
    test_frame[selected_columns].to_parquet(test_path, index=False)
    df[selected_columns].to_parquet(user_path, index=False)

    return output_dir / "all_data.parquet", train_path, val_path, test_path, user_path


def main() -> None:
    params = read_params()
    data_params = params.get("data", {})
    split_params = params.get("split", {})
    raw_csv = Path(data_params.get("raw_csv", "data/cyberbullying_tweets.csv"))
    output_dir = Path(data_params.get("output_dir", "data"))
    synthetic_users = int(data_params.get("synthetic_users", 5000))
    seed = int(split_params.get("seed", 42))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = build_dataset(raw_csv, output_dir, synthetic_users=synthetic_users, seed=seed)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
