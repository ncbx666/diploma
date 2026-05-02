from __future__ import annotations

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
