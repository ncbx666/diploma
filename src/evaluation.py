from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def evaluate_binary(y_true, y_pred_binary) -> tuple[dict[str, float], pd.DataFrame]:
    metrics = {
        "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred_binary)),
    }
    report = pd.DataFrame(classification_report(y_true, y_pred_binary, zero_division=0, output_dict=True)).T
    return metrics, report


def threshold_objective_value(y_true: Iterable[int], y_score: Iterable[float], threshold: float) -> float:
    """Return the F1 score produced by applying ``threshold`` to positive-class scores."""
    scores = np.nan_to_num(np.asarray(list(y_score), dtype=float), nan=-np.inf, posinf=1.0, neginf=-np.inf)
    y_pred = (scores >= float(threshold)).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def search_threshold(y_true: Iterable[int], y_score: Iterable[float]) -> tuple[float, float]:
    """Choose a validation threshold that maximizes F1.

    Candidate thresholds include a stable 0.01..0.99 grid plus the observed
    validation scores, so the search can both reproduce notebook-style grid
    tuning and hit exact decision boundaries on small smoke datasets.
    """
    scores = np.asarray(list(y_score), dtype=float)
    if scores.size == 0:
        return 0.5, 0.0

    finite_scores = scores[np.isfinite(scores)]

    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.clip(finite_scores, 0.0, 1.0),
                np.array([0.5], dtype=float),
            ]
        )
    )
    best_threshold = 0.5
    best_value = -1.0
    best_distance = float("inf")
    for threshold in candidates:
        value = threshold_objective_value(y_true, scores, float(threshold))
        distance = abs(float(threshold) - 0.5)
        if value > best_value or (value == best_value and (distance, threshold) < (best_distance, best_threshold)):
            best_threshold = float(threshold)
            best_value = float(value)
            best_distance = distance
    return best_threshold, best_value
