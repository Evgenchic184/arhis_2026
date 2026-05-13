from __future__ import annotations

from src.feature_store.offline import OfflineFeatureStore, generate_synthetic_user_features, generate_user_ids
from src.feature_store.schemas import CommentFeatureSnapshot, TrainingExample, UserFeatureSnapshot

__all__ = [
    "CommentFeatureSnapshot",
    "OfflineFeatureStore",
    "OnlineFeatureStore",
    "TrainingExample",
    "UserFeatureSnapshot",
    "generate_synthetic_user_features",
    "generate_user_ids",
]


def __getattr__(name: str):
    if name == "OnlineFeatureStore":
        from src.feature_store.online import OnlineFeatureStore

        return OnlineFeatureStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
