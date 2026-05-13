from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureConfigRead(BaseModel):
    text_column: str
    training_feature_columns: list[str]
    inference_feature_columns: list[str]
    online_user_feature_columns: list[str]
    version: int
    updated_at: str | None = None
    available_model_feature_columns: list[str] = Field(default_factory=list)
    available_user_feature_columns: list[str] = Field(default_factory=list)
    available_text_feature_columns: list[str] = Field(default_factory=list)


class FeatureConfigUpdate(BaseModel):
    text_column: str | None = None
    training_feature_columns: list[str] | None = None
    inference_feature_columns: list[str] | None = None
    online_user_feature_columns: list[str] | None = None
