import re
from typing import Dict, Any


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if len(w) > 2]
    
    return " ".join(words)


def extract_text_features(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        text = ""
    
    text_len = len(text)
    caps_count = sum(1 for c in text if c.isupper())
    
    return {
        'text_length': text_len,
        'caps_ratio': caps_count / max(text_len, 1),
        'has_url': int(bool(re.search(r'http\S+|www\S+', text))),
        'has_mention': int('@"' in text or '@' in text),
    }


def extract_features_from_prepared(text_prepared: str) -> Dict[str, Any]:
    if not isinstance(text_prepared, str):
        text_prepared = ""
    
    return {
        'text_length': len(text_prepared),
    }