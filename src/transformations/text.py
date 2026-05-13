from __future__ import annotations

import re
import string
from typing import Any

TOXIC_LEXICON = {
    "abuse",
    "asshole",
    "bitch",
    "dumb",
    "faggot",
    "fuck",
    "idiot",
    "kill",
    "nigger",
    "shit",
    "stupid",
    "trash",
    "ugly",
    "whore",
}


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = [word for word in text.split() if len(word) > 2]
    return " ".join(words)


def extract_text_features(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        text = ""

    tokens = text.split()
    text_length = len(text)
    token_count = len(tokens)
    caps_count = sum(1 for char in text if char.isupper())
    alpha_count = sum(1 for char in text if char.isalpha())
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    digit_count = sum(1 for char in text if char.isdigit())
    toxic_hits = sum(token.lower() in TOXIC_LEXICON for token in tokens)

    return {
        "text_length": text_length,
        "token_count": token_count,
        "caps_ratio": caps_count / max(text_length, 1),
        "alpha_ratio": alpha_count / max(text_length, 1),
        "punctuation_ratio": punctuation_count / max(text_length, 1),
        "digit_ratio": digit_count / max(text_length, 1),
        "avg_token_length": sum(len(token) for token in tokens) / max(token_count, 1),
        "unique_token_ratio": len(set(tokens)) / max(token_count, 1),
        "has_url": int(bool(re.search(r"http\S+|www\S+", text))),
        "has_mention": int(bool(re.search(r"@\w+", text))),
        "has_hashtag": int(bool(re.search(r"#\w+", text))),
        "num_exclamation_marks": text.count("!"),
        "num_question_marks": text.count("?"),
        "num_digits": digit_count,
        "repeated_char_sequences": len(re.findall(r"(.)\1{2,}", text.lower())),
        "toxic_keyword_hits": toxic_hits,
    }


def extract_required_text_features(text: str) -> dict[str, Any]:
    features = extract_text_features(text)
    return {
        "text_length": features["text_length"],
        "caps_ratio": features["caps_ratio"],
        "has_url": features["has_url"],
        "has_mention": features["has_mention"],
    }


def extract_features_from_prepared(text_prepared: str) -> dict[str, Any]:
    if not isinstance(text_prepared, str):
        text_prepared = ""

    tokens = text_prepared.split()
    return {
        "prepared_text_length": len(text_prepared),
        "prepared_token_count": len(tokens),
        "prepared_avg_token_length": sum(len(token) for token in tokens) / max(len(tokens), 1),
    }
