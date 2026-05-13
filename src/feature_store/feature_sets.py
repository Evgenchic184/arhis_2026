from __future__ import annotations

from collections.abc import Iterable

BASE_TEXT_COLUMN = "text_prepared"
BASE_LABEL_COLUMN = "cyberbullying_bin"
BASE_IDENTIFIER_COLUMN = "user_id"

BASE_SOURCE_COLUMNS = [
    "tweet_text",
    "cyberbullying_type",
]

BASE_TEXT_FEATURE_COLUMNS = [
    "text_length",
    "caps_ratio",
    "has_url",
    "has_mention",
]

EXTRA_TEXT_FEATURE_COLUMNS = [
    "token_count",
    "alpha_ratio",
    "punctuation_ratio",
    "digit_ratio",
    "avg_token_length",
    "unique_token_ratio",
    "num_exclamation_marks",
    "num_question_marks",
    "num_digits",
    "repeated_char_sequences",
    "toxic_keyword_hits",
]

AVAILABLE_TEXT_FEATURE_COLUMNS = BASE_TEXT_FEATURE_COLUMNS + EXTRA_TEXT_FEATURE_COLUMNS

BASE_USER_FEATURE_COLUMNS = [
    "is_new_user",
    "reputation_score",
    "reports_last_24h",
    "account_age_days",
]

EXTRA_USER_FEATURE_COLUMNS = [
    "reports_last_7d",
    "reports_last_30d",
    "deleted_comments_last_1d",
    "deleted_comments_last_7d",
    "deleted_comments_last_30d",
    "hidden_comments_last_1d",
    "hidden_comments_last_7d",
    "hidden_comments_last_30d",
    "comment_count_last_1d",
    "comment_count_last_7d",
    "comment_count_last_30d",
    "auto_action_count_last_30d",
    "manual_overrule_count_last_30d",
    "auto_action_rate_last_30d",
    "manual_overrule_rate_last_30d",
    "last_ml_confidence",
    "last_ml_verdict",
]

AVAILABLE_USER_FEATURE_COLUMNS = BASE_USER_FEATURE_COLUMNS + EXTRA_USER_FEATURE_COLUMNS

# Default numeric model inputs. These are the columns used before runtime overrides.
BASE_MODEL_FEATURE_COLUMNS = BASE_TEXT_FEATURE_COLUMNS + BASE_USER_FEATURE_COLUMNS

# All model-selectable features. `user_id` is intentionally excluded because it is an identifier,
# not a predictive feature. `last_ml_verdict` is kept out of this list because the current
# training pipeline does not encode categorical features.
AVAILABLE_MODEL_FEATURE_COLUMNS = AVAILABLE_TEXT_FEATURE_COLUMNS + [
    column for column in AVAILABLE_USER_FEATURE_COLUMNS if column != "last_ml_verdict"
]

# Columns that may appear in the materialized parquet files.
AVAILABLE_DATA_COLUMNS = (
    BASE_SOURCE_COLUMNS
    + [BASE_LABEL_COLUMN, BASE_TEXT_COLUMN, BASE_IDENTIFIER_COLUMN]
    + AVAILABLE_TEXT_FEATURE_COLUMNS
    + AVAILABLE_USER_FEATURE_COLUMNS
)

# Backwards-compatible alias for the base runtime feature set.
BASE_FEATURE_COLUMNS = list(BASE_MODEL_FEATURE_COLUMNS)

# Backwards-compatible alias for the materialized base set used in parquet files.
BASE_MODEL_INPUT_COLUMNS = [BASE_TEXT_COLUMN, BASE_IDENTIFIER_COLUMN] + BASE_MODEL_FEATURE_COLUMNS


def unique_preserving_order(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        result.append(column)
    return result


def validate_model_feature_columns(columns: Iterable[str]) -> list[str]:
    normalized = unique_preserving_order(columns)
    unknown = set(normalized) - set(AVAILABLE_MODEL_FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown feature columns: {sorted(unknown)}")
    return normalized


def validate_user_feature_columns(columns: Iterable[str]) -> list[str]:
    normalized = unique_preserving_order(columns)
    unknown = set(normalized) - set(AVAILABLE_USER_FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown user feature columns: {sorted(unknown)}")
    return normalized
