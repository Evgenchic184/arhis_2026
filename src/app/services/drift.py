from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def compute_psi(reference: Iterable[float], actual: Iterable[float], buckets: int = 10) -> float:
    ref = pd.Series(list(reference)).dropna()
    act = pd.Series(list(actual)).dropna()
    if ref.empty or act.empty:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, buckets + 1)
    edges = np.unique(ref.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)

    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / max(ref_counts.sum(), 1))
    act_pct = np.where(act_counts == 0, 1e-6, act_counts / max(act_counts.sum(), 1))
    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def compute_psi_from_bins(
    actual: Iterable[float],
    *,
    bin_edges: Iterable[float],
    reference_distribution: Iterable[float],
) -> float:
    actual_series = pd.Series(list(actual)).dropna()
    if actual_series.empty:
        return 0.0

    edges = np.asarray(list(bin_edges), dtype=float)
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    actual_counts, _ = np.histogram(actual_series, bins=edges)
    actual_pct = np.where(actual_counts == 0, 1e-6, actual_counts / max(actual_counts.sum(), 1))
    ref_pct = np.asarray(list(reference_distribution), dtype=float)
    if len(ref_pct) != len(actual_pct):
        return 0.0
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    return float(np.sum((actual_pct - ref_pct) * np.log(actual_pct / ref_pct)))


def tokenize_texts(texts: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        if not text:
            continue
        tokens.extend(str(text).lower().split())
    return tokens


def compute_new_token_share(texts: Iterable[str], vocabulary: set[str]) -> float:
    tokens = tokenize_texts(texts)
    if not tokens:
        return 0.0
    new_tokens = [token for token in tokens if token not in vocabulary]
    return float(len(new_tokens) / len(tokens))


def build_vocabulary(texts: Iterable[str]) -> set[str]:
    return set(tokenize_texts(texts))


def sampled_manual_accuracy(predicted_labels: Iterable[str], true_labels: Iterable[str]) -> float:
    predicted = list(predicted_labels)
    truth = list(true_labels)
    if not predicted or not truth:
        return 0.0
    if len(predicted) != len(truth):
        raise ValueError("Predicted and true label counts must match.")
    matches = sum(int(pred == true) for pred, true in zip(predicted, truth, strict=False))
    return float(matches / len(truth))
