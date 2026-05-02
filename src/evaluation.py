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


MIN_PRECISION = 0.60


def threshold_objective_value(recall_value: float, precision_value: float, min_precision: float = MIN_PRECISION) -> float:
    """Notebook-style objective: meet precision floor first, then maximize recall."""
    if precision_value >= min_precision:
        return 1.0 + float(recall_value) + 0.001 * float(precision_value)
    return float(precision_value) + 0.001 * float(recall_value)


def search_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    min_precision: float = MIN_PRECISION,
) -> tuple[float, dict[str, float]]:
    """Choose a validation threshold using the notebook precision-floor rule.

    Prefer thresholds where validation precision reaches ``min_precision`` and,
    among them, maximize validation recall. If none meets the floor, fall back
    to highest validation precision with recall/F1 as tie-breakers.
    """
    y_arr = np.asarray(list(y_true), dtype=int).reshape(-1)
    scores = np.nan_to_num(np.asarray(list(y_score), dtype=float).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    usable = min(y_arr.size, scores.size)
    if usable == 0:
        return 0.5, {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "objective": 0.0}
    y_arr = y_arr[:usable]
    scores = scores[:usable]

    candidates = np.unique(np.round(scores, 10))
    if candidates.size > 400:
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 400))
    candidates = np.unique(np.concatenate([candidates, [0.5]]))

    rows: list[dict[str, float]] = []
    for threshold in candidates:
        y_pred = (scores >= float(threshold)).astype(int)
        precision = float(precision_score(y_arr, y_pred, zero_division=0))
        recall = float(recall_score(y_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_arr, y_pred, zero_division=0))
        rows.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "objective": threshold_objective_value(recall, precision, min_precision),
        })

    candidates_df = pd.DataFrame(rows)
    eligible = candidates_df[candidates_df["precision"] >= min_precision]
    if not eligible.empty:
        best = eligible.sort_values(["recall", "precision", "f1", "threshold"], ascending=[False, False, False, True]).iloc[0]
    else:
        best = candidates_df.sort_values(["precision", "recall", "f1", "threshold"], ascending=[False, False, False, True]).iloc[0]
    info = {key: float(value) for key, value in best.to_dict().items()}
    return info["threshold"], info
