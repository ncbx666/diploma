#!/usr/bin/env python3
"""Kaggle-friendly entry point for potato late blight forecasting experiments.

The script keeps the notebook workflow simple and reproducible: read the Excel
file, create horizon targets, split by years, train selected models, save metrics
and artifacts, then immediately zip each experiment result folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models import build_model as build_transferred_model
from src.models import fit_estimator as fit_transferred_estimator
from src.models import predict_binary as predict_transferred_binary
from src.models import predict_scores as predict_transferred_scores
from src.models import suggest_params as suggest_transferred_params

try:  # optional on a clean Kaggle clone until requirements are installed
    import optuna
except Exception:  # pragma: no cover - depends on runtime
    optuna = None

try:  # plots are optional; skipped_plots.txt records missing support
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on runtime
    plt = None

RANDOM_STATE = 42
MIN_PRECISION = 0.60
DEFAULT_DATA_FILE = "Golitcino72-17d2_CLEAN.xlsx"
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/outputs")
SUPPORTED_MODELS = ("logreg", "svm", "rf", "gru", "tft", "blitecast", "xgboost", "catboost", "lightgbm", "arima", "sarima")
FEATURE_VARIANTS = (
    "baseline",
    "interaction_fe",
    "temporal_fe",
    "tomek_only",
    "tomek_interaction_fe",
    "tomek_temporal_fe",
    "tomek_interaction_temporal_fe",
)
FEATURE_VARIANT_DESCRIPTIONS = {
    "baseline": "Base predictors only.",
    "interaction_fe": "Base predictors plus two manual interaction features.",
    "temporal_fe": "Base predictors plus within-year lag, rolling, and trend features.",
    "tomek_only": "Base predictors with Tomek Links applied on the training split only.",
    "tomek_interaction_fe": "Interaction features plus Tomek Links on the training split only.",
    "tomek_temporal_fe": "Temporal features plus Tomek Links on the training split only.",
    "tomek_interaction_temporal_fe": "Interaction features, temporal features, and Tomek Links on the training split only.",
}
FEATURE_COLUMNS = [
    "day",
    "t_min",
    "t_max",
    "is_rain",
    "y2",
    "y1",
    "y2_y1",
    "precipitation",
    "t_avg",
    "cloudiness",
    "y3",
    "y4",
    "precipitation_t_gt_10",
    "t_gt_10",
]
ZERO_FILL_COLUMNS = {"is_rain", "precipitation", "precipitation_t_gt_10"}
INTERACTION_FEATURE_COLUMNS = ["fe_y4_cloud_interaction", "fe_cold_rain_index"]
TEMPORAL_SOURCE_COLUMNS = [
    "t_min",
    "t_max",
    "is_rain",
    "y1",
    "y2",
    "y2_y1",
    "precipitation",
    "t_avg",
    "cloudiness",
    "y3",
    "y4",
    "precipitation_t_gt_10",
    "t_gt_10",
]
TEMPORAL_LAG_STEPS = [1, 2, 3, 7]
TEMPORAL_ROLLING_WINDOWS = [3, 7]
TEMPORAL_ROLLING_STATS = ["mean", "std", "min", "max"]


@dataclass
class ExperimentResult:
    model: str
    variant: str
    horizon: int
    window_size: int
    f1: float
    output_dir: Path
    zip_path: Path
    threshold: float | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    val_f1: float | None = None
    val_accuracy: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run potato illness forecasting experiments.")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Path to Golitcino72-17d2_CLEAN.xlsx")
    parser.add_argument("--models", nargs="+", default=["logreg"], help="Models to run or 'all'.")
    parser.add_argument("--horizons", nargs="+", type=int, default=[2], help="Forecast horizons, e.g. --horizons 2 3")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[7], help="Window sizes, e.g. --window-sizes 5 7 9")
    parser.add_argument("--top-n", type=int, default=1, help="Keep only N best results per model ranked by F1.")
    parser.add_argument("--tune-trials", type=int, default=0, help="Number of Optuna trials per tunable experiment.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    parser.add_argument("--upload-dataset", action="store_true", help="Upload zips to a Kaggle Dataset after each experiment.")
    parser.add_argument("--dataset-slug", default=None, help="Kaggle Dataset slug, e.g. username/dataset-name.")
    return parser.parse_args()


def normalize_name(name: Any) -> str:
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def find_data_file(path_arg: str) -> Path:
    requested = Path(path_arg)
    candidates: list[Path] = []
    env_path = os.environ.get("DIPLOMA_EXCEL_FILE_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([requested, Path.cwd() / requested.name, Path("/kaggle/working") / requested.name])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{requested.name}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{requested.name} not found. Put it in the repo root, /kaggle/working, "
        "attach it as a Kaggle Dataset, or set DIPLOMA_EXCEL_FILE_PATH."
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    text = text.replace({"": np.nan, " ": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    text = text.astype(str).str.replace("б", "6", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def load_dataset(excel_path: Path) -> pd.DataFrame:
    book = pd.ExcelFile(excel_path, engine="openpyxl")
    year_sheets = [s for s in book.sheet_names if any(ch.isdigit() for ch in s) and len("".join(filter(str.isdigit, s))) == 4]
    frames: list[pd.DataFrame] = []
    if year_sheets:
        positional = [
            "year", "day", "t_min", "t_max", "is_rain", "target_favorable",
            "y2", "y1", "y2_y1", "precipitation", "t_avg", "cloudiness",
            "y3", "y4", "precipitation_t_gt_10", "t_gt_10",
        ]
        for sheet in year_sheets:
            year = int("".join(filter(str.isdigit, sheet)))
            raw = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl", nrows=92)
            raw = raw.iloc[:, : len(positional)].copy()
            raw.columns = positional[: raw.shape[1]]
            raw["year"] = year
            frames.append(raw)
    else:
        raw = pd.read_excel(excel_path, engine="openpyxl")
        raw.columns = [normalize_name(c) for c in raw.columns]
        if "unnamed_0" in raw.columns and "year" not in raw.columns:
            raw = raw.rename(columns={"unnamed_0": "year"})
        raw = raw.rename(columns={"t_10": "t_gt_10", "t>10": "t_gt_10", "y2_y1": "y2_y1"})
        frames.append(raw)

    df = pd.concat(frames, ignore_index=True)
    df.columns = [normalize_name(c) for c in df.columns]
    rename_map = {"t_10": "t_gt_10", "t_10_": "t_gt_10", "target_favorable": "target_favorable"}
    df = df.rename(columns=rename_map)
    if "year" not in df.columns or "day" not in df.columns or "target_favorable" not in df.columns:
        raise ValueError("Data must contain year/day/Target_Favorable columns after normalization.")
    for column in df.columns:
        df[column] = clean_numeric(df[column])
    df = df.dropna(subset=["year", "day", "target_favorable"]).copy()
    df["year"] = df["year"].round().astype(int)
    df["day"] = df["day"].round().astype(int)
    df["target_favorable"] = df["target_favorable"].round().clip(0, 1).astype(int)
    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = np.nan
    return df.sort_values(["year", "day"]).reset_index(drop=True)


def add_targets(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    for horizon in horizons:
        out[f"target_h{horizon}"] = out.groupby("year")["target_favorable"].shift(-horizon)
    return out


def split_years(df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    years = sorted(int(y) for y in df["year"].dropna().unique())
    if len(years) < 3:
        raise ValueError("Need at least 3 years for train/validation/test split.")
    test_count = min(5, max(1, len(years) // 5))
    val_count = min(7, max(1, (len(years) - test_count) // 5))
    train_years = years[: -(val_count + test_count)]
    val_years = years[-(val_count + test_count) : -test_count]
    test_years = years[-test_count:]
    if not train_years:
        train_years, val_years, test_years = years[:-2], [years[-2]], [years[-1]]
    return train_years, val_years, test_years


def prepare_features(df: pd.DataFrame, target_col: str, window_size: int, variant: str) -> tuple[pd.DataFrame, list[str]]:
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    work = df.copy().sort_values(["year", "day"]).reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        if col in ZERO_FILL_COLUMNS:
            work[col] = work[col].fillna(0.0)
        work[col] = work.groupby("year")[col].transform(lambda s: s.ffill().bfill())
        work[col] = work[col].fillna(work[col].median())
    feature_cols = FEATURE_COLUMNS.copy()

    if "interaction" in variant:
        work["fe_y4_cloud_interaction"] = work["y4"] * work["cloudiness"]
        work["fe_cold_rain_index"] = work["is_rain"] * (35 - work["t_avg"])
        feature_cols.extend(INTERACTION_FEATURE_COLUMNS)

    if "temporal" in variant:
        grouped = work.groupby("year", sort=False)
        for col in TEMPORAL_SOURCE_COLUMNS:
            for lag in TEMPORAL_LAG_STEPS:
                lag_col = f"{col}_lag_{lag}"
                work[lag_col] = grouped[col].shift(lag)
                feature_cols.append(lag_col)
            for rolling_window in TEMPORAL_ROLLING_WINDOWS:
                rolled = grouped[col].rolling(rolling_window, min_periods=rolling_window)
                for stat in TEMPORAL_ROLLING_STATS:
                    roll_col = f"{col}_roll_{rolling_window}_{stat}"
                    if stat == "mean":
                        values = rolled.mean()
                    elif stat == "std":
                        values = rolled.std()
                    elif stat == "min":
                        values = rolled.min()
                    elif stat == "max":
                        values = rolled.max()
                    else:
                        raise ValueError(stat)
                    work[roll_col] = values.reset_index(level=0, drop=True)
                    feature_cols.append(roll_col)
            trend_col = f"{col}_trend_3_vs_prev3"
            current_mean = grouped[col].rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
            previous_mean = (
                work.groupby("year")[col]
                .shift(3)
                .groupby(work["year"])
                .rolling(3, min_periods=3)
                .mean()
                .reset_index(level=0, drop=True)
            )
            work[trend_col] = current_mean - previous_mean
            feature_cols.append(trend_col)

    feature_cols = list(dict.fromkeys(feature_cols))
    work = work.dropna(subset=[target_col]).copy()
    work[target_col] = work[target_col].round().clip(0, 1).astype(int)
    return work, feature_cols


def apply_tomek_if_needed(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    variant: str,
) -> tuple[pd.DataFrame, pd.Series, str | None]:
    if "tomek" not in variant:
        return x_train, y_train, None
    try:
        from imblearn.under_sampling import TomekLinks
    except ImportError as exc:
        raise ImportError("Tomek feature variants require imbalanced-learn from requirements.txt.") from exc

    sampler = TomekLinks()
    sample_frame = x_train.replace([np.inf, -np.inf], np.nan)
    medians = sample_frame.median(numeric_only=True).fillna(0)
    sample_frame = sample_frame.fillna(medians).fillna(0)
    x_resampled, y_resampled = sampler.fit_resample(sample_frame, y_train)
    removed = len(x_train) - len(x_resampled)
    note = f"Tomek Links applied to training split only; removed {removed} rows."
    return pd.DataFrame(x_resampled, columns=x_train.columns), pd.Series(y_resampled, name=y_train.name), note


def choose_models(raw_models: list[str]) -> list[str]:
    requested = [m.lower() for m in raw_models]
    if "all" in requested:
        return list(SUPPORTED_MODELS)
    unknown = sorted(set(requested) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Supported: all, {', '.join(SUPPORTED_MODELS)}")
    return requested


def build_estimator(model: str, params: dict[str, Any] | None = None) -> Any:
    params = params or {}
    if model == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=params.get("C", 1.0), max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
    if model == "svm":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=params.get("C", 1.0), gamma=params.get("gamma", "scale"), probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
    if model == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=params.get("n_estimators", 200), max_depth=params.get("max_depth"), class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ])
    if model in {"xgboost", "catboost", "lightgbm"}:
        return build_transferred_model(model, params)
    raise ValueError(f"Model {model} is not a tabular sklearn estimator.")


def threshold_objective_value(recall_value: float, precision_value: float) -> float:
    if precision_value >= MIN_PRECISION:
        return 1.0 + recall_value + 0.001 * precision_value
    return precision_value + 0.001 * recall_value


def search_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    if y_true.size == 0 or scores.size == 0:
        raise ValueError("Cannot search threshold on an empty validation split.")
    if y_true.size != scores.size:
        usable = min(y_true.size, scores.size)
        y_true = y_true[:usable]
        scores = scores[:usable]
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    unique_scores = np.unique(np.round(scores, 10))
    if unique_scores.size == 1:
        unique_scores = np.array([unique_scores[0]])
    if unique_scores.size > 400:
        unique_scores = np.quantile(scores, np.linspace(0.0, 1.0, 400))
    unique_scores = np.unique(np.concatenate([unique_scores, [0.5]]))

    candidates: list[dict[str, float]] = []
    for threshold in unique_scores:
        y_pred = (scores >= threshold).astype(int)
        recall_value = recall_score(y_true, y_pred, zero_division=0)
        precision_value = precision_score(y_true, y_pred, zero_division=0)
        f1_value = f1_score(y_true, y_pred, zero_division=0)
        candidates.append({
            "threshold": float(threshold),
            "recall": float(recall_value),
            "precision": float(precision_value),
            "f1": float(f1_value),
            "objective": float(threshold_objective_value(recall_value, precision_value)),
        })

    candidates_df = pd.DataFrame(candidates)
    eligible = candidates_df[candidates_df["precision"] >= MIN_PRECISION]
    if not eligible.empty:
        best_row = eligible.sort_values(["recall", "precision", "f1", "threshold"], ascending=[False, False, False, True]).iloc[0]
    else:
        best_row = candidates_df.sort_values(["precision", "recall", "f1", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return {key: float(value) for key, value in best_row.to_dict().items()}


def validation_metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int).reshape(-1)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    usable = min(y_true_arr.size, scores_arr.size)
    y_true_arr = y_true_arr[:usable]
    scores_arr = scores_arr[:usable]
    y_pred = (scores_arr >= threshold).astype(int)
    return {
        "val_precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "val_recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "val_f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "val_accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }


def tune_params(model: str, x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series, trials: int) -> dict[str, Any]:
    if trials <= 0 or optuna is None or model not in {"logreg", "svm", "rf", "xgboost", "catboost", "lightgbm"}:
        return {}

    def objective(trial: Any) -> float:
        if model == "logreg":
            params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
        elif model == "svm":
            params = {"C": trial.suggest_float("C", 0.1, 20.0, log=True), "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])}
        elif model == "rf":
            params = {"n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 2, 12)}
        else:
            params = suggest_transferred_params(trial, model, y_train)
        estimator = build_estimator(model, params)
        if model in {"xgboost", "catboost", "lightgbm"}:
            fit_transferred_estimator(model, estimator, x_train, y_train, x_val, y_val)
            pred = predict_transferred_binary(model, estimator, x_val)
        else:
            estimator.fit(x_train, y_train)
            if model == "logreg":
                val_scores = predict_scores(estimator, x_val)
                if val_scores is None:
                    pred = estimator.predict(x_val)
                    return f1_score(y_val, pred, zero_division=0)
                threshold_info = search_threshold(y_val.to_numpy(dtype=int), val_scores)
                return threshold_objective_value(threshold_info["recall"], threshold_info["precision"])
            pred = estimator.predict(x_val)
        return f1_score(y_val, pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params)


def predict_scores(estimator: Any, x_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x_test)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(x_test)
        return np.asarray(score).reshape(-1)
    return None


def run_arima_like(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    # Conservative ARIMA-style baseline: use recent target rolling mean by year, no artificial dependency fallback.
    # If statsmodels is installed, users can extend this without changing the train.py contract.
    history_rate = float(train_df[target_col].mean()) if len(train_df) else 0.5
    scores = []
    for _, group in test_df.sort_values(["year", "day"]).groupby("year"):
        rolling = group[target_col].shift(1).expanding(min_periods=1).mean().fillna(history_rate)
        scores.extend(rolling.to_numpy(dtype=float).tolist())
    y_true = test_df.sort_values(["year", "day"])[target_col].astype(int).to_numpy()
    y_score = np.asarray(scores, dtype=float)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_pred, y_score, {"note": "ARIMA-style rolling target-rate baseline from notebook time-series family"}



def run_gru_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
    """Train a small real GRU classifier on tabular weather windows.

    The source notebook used year-aware sequences.  This CLI keeps the first
    version Kaggle-friendly by treating each engineered feature row as a
    one-step sequence, while still using an actual torch.nn.GRU model.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("GRU requires torch. Install requirements.txt before running --models gru.") from exc

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    fit_x = pd.concat([x_train, x_val], ignore_index=True) if not x_val.empty else x_train
    fit_y = pd.concat([y_train, y_val], ignore_index=True) if not y_val.empty else y_train
    train_arr = scaler.fit_transform(imputer.fit_transform(fit_x)).astype("float32")
    test_arr = scaler.transform(imputer.transform(x_test)).astype("float32")
    y_arr = fit_y.to_numpy(dtype="float32")

    torch.manual_seed(RANDOM_STATE)

    class GRUClassifier(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=input_size, hidden_size=16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, features):
            _, hidden = self.gru(features.unsqueeze(1))
            return self.head(hidden[-1]).squeeze(1)

    model = GRUClassifier(train_arr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.tensor(train_arr), torch.tensor(y_arr))
    loader = DataLoader(ds, batch_size=min(64, max(1, len(ds))), shuffle=True)
    model.train()
    for _ in range(max(3, int(epochs) if epochs > 0 else 5)):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_arr))
        y_score = torch.sigmoid(logits).cpu().numpy().astype(float)
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score, model, {"epochs": max(3, int(epochs) if epochs > 0 else 5), "hidden_size": 16}

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_config(path: Path, config: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        elif isinstance(value, dict):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_placeholder_png(path: Path) -> None:
    """Write a tiny valid PNG when matplotlib is unavailable.

    classification_report.png is a mandatory artifact for every experiment,
    but Kaggle smoke-test environments may not have matplotlib installed yet.
    Other optional plots still follow skipped_plots.txt instead of fake
    fallbacks.
    """
    width, height = 16, 16
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00")
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)




def write_confusion_png(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Write a dependency-free confusion-matrix heatmap PNG.

    This keeps confusion_matrix.png mandatory even in minimal Kaggle smoke
    environments where matplotlib has not been installed yet.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    max_value = float(matrix.max()) or 1.0
    cell = 32
    width = height = cell * 2
    rows = []
    for y in range(height):
        row = bytearray(b"\x00")
        for x in range(width):
            value = matrix[y // cell, x // cell] / max_value
            blue = 120 + int(135 * value)
            red_green = 255 - int(180 * value)
            row.extend([red_green, red_green, blue])
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00")
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def safe_folder_name(output_root: Path, model: str, f1_value: float, variant: str | None = None) -> Path:
    score = int(round(max(0.0, min(1.0, f1_value)) * 100))
    prefix = f"{model}_{variant}" if variant else model
    base = output_root / f"{prefix}_{score:02d}"
    if not base.exists():
        return base
    suffix = 2
    while (output_root / f"{prefix}_{score:02d}_{suffix}").exists():
        suffix += 1
    return output_root / f"{prefix}_{score:02d}_{suffix}"


def save_plots(result_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None, estimator: Any, feature_cols: list[str]) -> None:
    skipped: list[str] = []
    if plt is None:
        write_confusion_png(result_dir / "confusion_matrix.png", y_true, y_pred)
        skipped.extend([
            "roc_curve.png: matplotlib is not installed",
            "precision_recall_curve.png: matplotlib is not installed",
            "feature_importance.png: matplotlib is not installed",
        ])
    else:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(result_dir / "confusion_matrix.png")
        plt.close(fig)

        if y_score is None or len(np.unique(y_true)) < 2:
            skipped.append("roc_curve.png: y_score is unavailable or y_true has one class")
            skipped.append("precision_recall_curve.png: y_score is unavailable or y_true has one class")
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr, tpr)
            ax.set_title("ROC curve")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            fig.tight_layout()
            fig.savefig(result_dir / "roc_curve.png")
            plt.close(fig)

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(recall, precision)
            ax.set_title("Precision-recall curve")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            fig.tight_layout()
            fig.savefig(result_dir / "precision_recall_curve.png")
            plt.close(fig)

        model_obj = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else estimator
        importances = getattr(model_obj, "feature_importances_", None)
        if importances is None:
            skipped.append("feature_importance.png: model does not naturally expose feature_importances_")
        else:
            order = np.argsort(importances)[-20:]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(np.asarray(feature_cols)[order], np.asarray(importances)[order])
            ax.set_title("Feature importance")
            fig.tight_layout()
            fig.savefig(result_dir / "feature_importance.png")
            plt.close(fig)
    if skipped:
        (result_dir / "skipped_plots.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")


def zip_result(result_dir: Path, output_root: Path) -> Path:
    zips_dir = output_root / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zips_dir / f"{result_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(result_dir.parent))
    return zip_path


def ensure_persistent_copy(zip_path: Path, output_root: Path, dataset_slug: str | None) -> Path:
    persistent_dir = output_root / "persistent_dataset"
    persistent_dir.mkdir(parents=True, exist_ok=True)
    copied = persistent_dir / zip_path.name
    shutil.copy2(zip_path, copied)
    if dataset_slug:
        metadata = {"id": dataset_slug, "title": dataset_slug.split("/", 1)[-1]}
        (persistent_dir / "dataset-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return copied


def maybe_upload(zip_path: Path, output_root: Path, upload: bool, dataset_slug: str | None) -> None:
    if not upload:
        warnings.warn(
            f"Zip saved locally at {zip_path}. Files under /kaggle/working may disappear after the Kaggle runtime stops. "
            "Use --upload-dataset --dataset-slug username/dataset-name to persist them in a Kaggle Dataset.",
            RuntimeWarning,
        )
        return
    if not dataset_slug:
        raise ValueError("--dataset-slug is required when --upload-dataset is set.")
    persistent_dir = ensure_persistent_copy(zip_path, output_root, dataset_slug)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(persistent_dir.parent), "-m", "new experiment results"],
                check=True,
            )
            return
        except Exception as exc:  # pragma: no cover - requires Kaggle CLI/credentials
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Kaggle Dataset upload failed after 3 attempts: {last_error}")


def run_one(
    model: str,
    variant: str,
    horizon: int,
    window_size: int,
    df: pd.DataFrame,
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    args: argparse.Namespace,
) -> ExperimentResult | None:
    target_col = f"target_h{horizon}"
    prepared, feature_cols = prepare_features(df, target_col, window_size, variant)
    train_df = prepared[prepared["year"].isin(train_years)].copy()
    val_df = prepared[prepared["year"].isin(val_years)].copy()
    test_df = prepared[prepared["year"].isin(test_years)].copy()
    if train_df.empty or test_df.empty:
        print(f"SKIP {model} variant={variant} h{horizon} w{window_size}: empty train/test split")
        return None

    if model in {"gru", "tft"}:
        result_dir = safe_folder_name(args.output_dir, model, 0.0, variant)
        result_dir.mkdir(parents=True, exist_ok=True)
        reason = (
            "gru skipped: GRU training requires the PyTorch sequence-model path from the notebook; "
            "no artificial fallback was created."
            if model == "gru"
            else "tft skipped: Temporal Fusion Transformer requires pytorch-forecasting/lightning and GPU-oriented setup; "
            "no artificial fallback was created."
        )
        (result_dir / "skipped_model.txt").write_text(reason + "\n", encoding="utf-8")
        write_config(result_dir / "config_used.yaml", vars(args) | {"model": model, "variant": variant, "horizon": horizon, "window_size": window_size})
        zip_path = zip_result(result_dir, args.output_dir)
        maybe_upload(zip_path, args.output_dir, args.upload_dataset, args.dataset_slug)
        return ExperimentResult(model, variant, horizon, window_size, 0.0, result_dir, zip_path)

    x_train, y_train = train_df[feature_cols], train_df[target_col].astype(int)
    x_val, y_val = val_df[feature_cols], val_df[target_col].astype(int)
    x_test, y_test = test_df[feature_cols], test_df[target_col].astype(int)
    tomek_note = None
    if model not in {"arima", "sarima"}:
        x_train, y_train, tomek_note = apply_tomek_if_needed(x_train, y_train, variant)

    best_params: dict[str, Any] = {}
    selected_threshold: float | None = None
    threshold_info: dict[str, float] = {}
    val_metric_values: dict[str, float] = {}
    if model in {"logreg", "svm", "rf", "xgboost", "catboost", "lightgbm"}:
        best_params = tune_params(model, x_train, y_train, x_val, y_val, args.tune_trials)
        estimator = build_estimator(model, best_params)
        if model == "logreg" and not x_val.empty:
            estimator.fit(x_train, y_train)
            val_score = predict_scores(estimator, x_val)
            if val_score is None:
                val_pred = estimator.predict(x_val).astype(int)
                y_pred = estimator.predict(x_test).astype(int)
                y_score = predict_scores(estimator, x_test)
                selected_threshold = 0.5
                threshold_info = {
                    "threshold": selected_threshold,
                    "recall": float(recall_score(y_val, val_pred, zero_division=0)),
                    "precision": float(precision_score(y_val, val_pred, zero_division=0)),
                    "f1": float(f1_score(y_val, val_pred, zero_division=0)),
                    "objective": 0.0,
                }
                val_metric_values = {
                    "val_precision": threshold_info["precision"],
                    "val_recall": threshold_info["recall"],
                    "val_f1": threshold_info["f1"],
                    "val_accuracy": float(accuracy_score(y_val, val_pred)),
                }
            else:
                threshold_info = search_threshold(y_val.to_numpy(dtype=int), val_score)
                selected_threshold = threshold_info["threshold"]
                y_score = predict_scores(estimator, x_test)
                if y_score is None:
                    y_pred = estimator.predict(x_test).astype(int)
                else:
                    y_pred = (np.asarray(y_score, dtype=float) >= selected_threshold).astype(int)
                val_metric_values = validation_metrics(y_val, val_score, selected_threshold)
        else:
            fit_x = pd.concat([x_train, x_val], ignore_index=True) if not x_val.empty else x_train
            fit_y = pd.concat([y_train, y_val], ignore_index=True) if not y_val.empty else y_train
            if model in {"xgboost", "catboost", "lightgbm"}:
                fit_transferred_estimator(model, estimator, fit_x, fit_y, x_val if not x_val.empty else None, y_val if not x_val.empty else None)
                y_pred = predict_transferred_binary(model, estimator, x_test)
                y_score = predict_transferred_scores(model, estimator, x_test)
            else:
                estimator.fit(fit_x, fit_y)
                y_pred = estimator.predict(x_test).astype(int)
                y_score = predict_scores(estimator, x_test)
    elif model == "blitecast":
        aligned = test_df.sort_values(["year", "day"])
        precipitation = aligned.get("precipitation", pd.Series(0, index=aligned.index)).fillna(0)
        t_avg = aligned.get("t_avg", pd.Series(0, index=aligned.index)).fillna(0)
        cloudiness = aligned.get("cloudiness", pd.Series(0, index=aligned.index)).fillna(0)
        y_score = (
            0.45 * (precipitation > 0).astype(float)
            + 0.30 * t_avg.between(12, 20).astype(float)
            + 0.25 * (cloudiness >= 6).astype(float)
        ).to_numpy()
        y_pred = (y_score >= 0.5).astype(int)
        y_test = aligned[target_col].astype(int)
        estimator = None
    elif model in {"arima", "sarima"}:
        y_true_arima, y_pred, y_score, best_params = run_arima_like(pd.concat([train_df, val_df]), test_df, target_col)
        if model == "sarima":
            best_params["seasonal_note"] = "SARIMA-compatible baseline; full statsmodels SARIMAX is available through src.models when statsmodels is installed"
        y_test = pd.Series(y_true_arima)
        estimator = None
    else:
        raise ValueError(model)

    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_test, y_pred))
    result_dir = safe_folder_name(args.output_dir, model, f1, variant)
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics = [{
        "model": model,
        "variant": variant,
        "variant_description": FEATURE_VARIANT_DESCRIPTIONS[variant],
        "horizon": horizon,
        "window_size": window_size,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "threshold": selected_threshold,
        "threshold_objective": threshold_info.get("objective"),
        "val_precision": val_metric_values.get("val_precision"),
        "val_recall": val_metric_values.get("val_recall"),
        "val_f1": val_metric_values.get("val_f1"),
        "val_accuracy": val_metric_values.get("val_accuracy"),
        "feature_count": len(feature_cols),
        "tomek_note": tomek_note or "",
        "best_params": json.dumps(best_params, ensure_ascii=False),
    }]
    write_csv(result_dir / "metrics.csv", metrics)
    if selected_threshold is not None:
        threshold_payload = {
            "model": model,
            "variant": variant,
            "horizon": horizon,
            "window_size": window_size,
            "selected_threshold": selected_threshold,
            "min_precision": MIN_PRECISION,
            "validation": val_metric_values,
            "selection": threshold_info,
        }
        (result_dir / "validation_threshold.json").write_text(
            json.dumps(threshold_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    pred_rows = test_df[["year", "day"]].copy().reset_index(drop=True)
    pred_rows["y_true"] = np.asarray(y_test, dtype=int)
    pred_rows["y_pred_binary"] = np.asarray(y_pred, dtype=int)
    if y_score is not None:
        pred_rows["y_score"] = np.asarray(y_score, dtype=float)
    pred_rows.to_csv(result_dir / "predictions.csv", index=False)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(result_dir / "classification_report.csv")
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.table(cellText=pd.DataFrame(report).T.round(3).reset_index().values, colLabels=["class", *pd.DataFrame(report).T.columns], loc="center")
        fig.tight_layout()
        fig.savefig(result_dir / "classification_report.png")
        plt.close(fig)
    else:
        write_placeholder_png(result_dir / "classification_report.png")
    write_config(
        result_dir / "config_used.yaml",
        vars(args)
        | {
            "model": model,
            "variant": variant,
            "variant_description": FEATURE_VARIANT_DESCRIPTIONS[variant],
            "horizon": horizon,
            "window_size": window_size,
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "tomek_note": tomek_note or "",
            "best_params": best_params,
            "threshold": selected_threshold,
            "threshold_objective": threshold_info.get("objective"),
            **val_metric_values,
        },
    )
    (result_dir / "train_val_test_years.json").write_text(
        json.dumps({"train": train_years, "validation": val_years, "test": test_years}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_plots(result_dir, np.asarray(y_test, dtype=int), np.asarray(y_pred, dtype=int), y_score, estimator, feature_cols)
    zip_path = zip_result(result_dir, args.output_dir)
    maybe_upload(zip_path, args.output_dir, args.upload_dataset, args.dataset_slug)
    print(f"DONE {model} variant={variant} horizon={horizon} window={window_size} f1={f1:.3f} -> {result_dir} zip={zip_path}")
    return ExperimentResult(
        model,
        variant,
        horizon,
        window_size,
        f1,
        result_dir,
        zip_path,
        threshold=selected_threshold,
        val_precision=val_metric_values.get("val_precision"),
        val_recall=val_metric_values.get("val_recall"),
        val_f1=val_metric_values.get("val_f1"),
        val_accuracy=val_metric_values.get("val_accuracy"),
    )


def enforce_top_n(results: list[ExperimentResult], top_n: int) -> list[ExperimentResult]:
    if top_n <= 0:
        return results
    by_model: dict[str, list[ExperimentResult]] = {}
    for result in results:
        by_model.setdefault(result.model, []).append(result)
    kept_results: list[ExperimentResult] = []
    for model_results in by_model.values():
        keep = sorted(model_results, key=lambda r: r.f1, reverse=True)[:top_n]
        kept_results.extend(keep)
        keep_dirs = {r.output_dir for r in keep}
        keep_zips = {r.zip_path for r in keep}
        for result in model_results:
            if result.output_dir not in keep_dirs and result.output_dir.exists():
                shutil.rmtree(result.output_dir)
            if result.zip_path not in keep_zips and result.zip_path.exists():
                result.zip_path.unlink()
    return sorted(kept_results, key=lambda row: (row.model, -row.f1))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = choose_models(args.models)
    data_path = find_data_file(args.data)
    df = add_targets(load_dataset(data_path), args.horizons)
    train_years, val_years, test_years = split_years(df)
    print(f"Data: {data_path} rows={len(df)} years={df['year'].nunique()}")
    print(f"Train years: {train_years}")
    print(f"Validation years: {val_years}")
    print(f"Test years: {test_years}")
    if args.tune_trials > 0 and optuna is None:
        warnings.warn("Optuna is not installed; running default parameters.", RuntimeWarning)

    results: list[ExperimentResult] = []
    for model in models:
        for horizon in args.horizons:
            for window_size in args.window_sizes:
                for variant in FEATURE_VARIANTS:
                    result = run_one(model, variant, horizon, window_size, df, train_years, val_years, test_years, args)
                    if result is not None:
                        results.append(result)
    results = enforce_top_n(results, args.top_n)
    summary_rows = [r.__dict__ | {"output_dir": str(r.output_dir), "zip_path": str(r.zip_path)} for r in results]
    write_csv(args.output_dir / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
