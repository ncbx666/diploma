#!/usr/bin/env python3
"""Weather-only forecasting benchmark for Kaggle.

This entry point intentionally does not train disease classifiers and never uses
Target_Favorable, y*, t>10, or precipitation_t>10 columns as inputs or targets.
It forecasts raw weather variables from W days of raw weather history.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

try:
    import optuna
except Exception:  # pragma: no cover - depends on Kaggle environment
    optuna = None

try:
    import joblib
except Exception:  # pragma: no cover - artifact convenience only
    joblib = None


RANDOM_STATE = 42
DEFAULT_DATA_FILE = "Golitcino72-17d2_CLEAN.xlsx"
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/weather_outputs")
FORECAST_VARIABLES = ["t_min", "t_max", "t_avg", "precipitation", "is_rain", "cloudiness"]
RAIN_TARGET = "is_rain"
LEADS = {"lead_1": 1, "lead_2": 2}
SUPPORTED_MODELS = ("catboost_hybrid", "lightgbm_hybrid", "chronos2")
SUPPORTED_WINDOWS = (7, 11)
FORBIDDEN_COLUMNS = {
    "target_favorable",
    "target",
    "y1",
    "y2",
    "y2_y1",
    "y3",
    "y4",
    "precipitation_t_gt_10",
    "t_gt_10",
    "target_h1",
    "target_h2",
}


@dataclass
class FittedHead:
    model: Any
    imputer: SimpleImputer | None
    blend_weight: float
    threshold: float | None = None


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (np.full(len(x), self.probability, dtype=float) >= 0.5).astype(int)


@dataclass
class ModelRun:
    model: str
    window_size: int
    run_dir: Path
    zip_path: Path
    metrics: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather-only t+1/t+2 forecasting benchmark.")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Excel/CSV data path. Kaggle input paths are searched by basename.")
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS), help="Models to run: catboost_hybrid lightgbm_hybrid chronos2 or all.")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[7, 11], help="History lengths. Allowed values: 7 and 11.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--val-year-count", type=int, default=6, help="Number of most-recent pre-test years for validation.")
    parser.add_argument("--test-year-count", type=int, default=6, help="Number of most-recent years for final test.")
    parser.add_argument("--tune-trials", type=int, default=0, help="Optuna trials per model/target/lead. Use 0 for fast Kaggle smoke tests.")
    parser.add_argument("--iterations", type=int, default=800, help="Default boosting iterations before early stopping.")
    parser.add_argument("--early-stopping-rounds", type=int, default=80, help="Early stopping rounds for boosting models.")
    parser.add_argument("--chronos-model-id", default="amazon/chronos-2", help="Hugging Face model id for Chronos-2.")
    parser.add_argument("--chronos-device-map", default="auto", help="Chronos-2 device_map, e.g. auto, cuda, cpu.")
    parser.add_argument("--chronos-batch-size", type=int, default=64, help="Rolling-window batch size for Chronos-2 inference.")
    parser.add_argument("--upload-dataset", action="store_true", help="Copy zipped artifacts to a Kaggle Dataset staging dir.")
    parser.add_argument("--dataset-slug", default=None, help="Kaggle Dataset slug when --upload-dataset is set.")
    return parser.parse_args()


def normalize_name(name: object) -> str:
    text = str(name).strip().lower().replace(">", "_gt_").replace("-", "_").replace(" ", "_")
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def clean_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    text = text.replace({"": np.nan, " ": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan, "-": np.nan})
    text = text.astype(str).str.replace("б", "6", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


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


def _read_excel_year_sheets(excel_path: Path) -> pd.DataFrame:
    book = pd.ExcelFile(excel_path, engine="openpyxl")
    year_sheets = [s for s in book.sheet_names if any(ch.isdigit() for ch in str(s)) and len("".join(filter(str.isdigit, str(s)))) == 4]
    frames: list[pd.DataFrame] = []
    positional = [
        "year",
        "day",
        "t_min",
        "t_max",
        "is_rain",
        "target_favorable",
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
    for sheet in year_sheets:
        year = int("".join(filter(str.isdigit, str(sheet))))
        raw = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl", nrows=92)
        raw = raw.iloc[:, : len(positional)].copy()
        raw.columns = positional[: raw.shape[1]]
        raw["year"] = year
        frames.append(raw)
    if not frames:
        raw = pd.read_excel(excel_path, engine="openpyxl")
        raw.columns = [normalize_name(c) for c in raw.columns]
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def load_weather_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load only year, day, and raw weather variables from Excel or CSV."""
    path = Path(data_path)
    if path.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(path)
        raw.columns = [normalize_name(c) for c in raw.columns]
    else:
        raw = _read_excel_year_sheets(path)
        raw.columns = [normalize_name(c) for c in raw.columns]
    raw = raw.rename(columns={"t_10": "t_gt_10", "t_10_": "t_gt_10"})
    required = ["year", "day", *FORECAST_VARIABLES]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Weather benchmark requires columns {required}; missing: {missing}")

    data = raw[required].copy()
    for column in required:
        data[column] = clean_numeric(data[column])
    data = data.dropna(subset=["year", "day"]).copy()
    data["year"] = data["year"].round().astype(int)
    data["day"] = data["day"].round().astype(int)
    data[RAIN_TARGET] = data[RAIN_TARGET].fillna(0).round().clip(0, 1)
    data = data.sort_values(["year", "day"]).reset_index(drop=True)
    return data


def split_years(df: pd.DataFrame, val_year_count: int, test_year_count: int) -> tuple[list[int], list[int], list[int]]:
    years = sorted(int(y) for y in df["year"].dropna().unique())
    if len(years) < 3:
        raise ValueError("At least three years are required for train/validation/test splitting.")
    if len(years) <= val_year_count + test_year_count:
        test_year_count = max(1, math.ceil(len(years) * 0.2))
        val_year_count = max(1, math.ceil((len(years) - test_year_count) * 0.2))
    train_end = len(years) - val_year_count - test_year_count
    if train_end <= 0:
        train_end = 1
    val_end = len(years) - test_year_count
    return years[:train_end], years[train_end:val_end], years[val_end:]


def validate_window_sizes(window_sizes: Iterable[int]) -> list[int]:
    windows = sorted(set(int(w) for w in window_sizes))
    invalid = [w for w in windows if w not in SUPPORTED_WINDOWS]
    if invalid:
        raise ValueError(f"Weather benchmark supports W=7 or W=11 only; invalid: {invalid}")
    return windows


def validate_models(models: Iterable[str]) -> list[str]:
    requested = [m.strip().lower().replace("-", "_") for m in models]
    if "all" in requested:
        return list(SUPPORTED_MODELS)
    unknown = sorted(set(requested) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unknown weather models: {unknown}. Supported: all, {', '.join(SUPPORTED_MODELS)}")
    return list(dict.fromkeys(requested))


def target_column(lead_name: str, variable: str) -> str:
    return f"{lead_name}_{variable}"


def baseline_column(lead_name: str, variable: str) -> str:
    return f"{lead_name}_{variable}_persistence"


def make_weather_windows(df: pd.DataFrame, window_size: int) -> tuple[pd.DataFrame, list[str]]:
    """Create supervised rows from W days ending at t to targets at t+1 and t+2."""
    if window_size not in SUPPORTED_WINDOWS:
        raise ValueError("window_size must be 7 or 11")
    rows: list[dict[str, float | int]] = []
    for year, group in df.sort_values(["year", "day"]).groupby("year", sort=False):
        group = group.reset_index(drop=True)
        weather = group[FORECAST_VARIABLES]
        for idx in range(window_size - 1, len(group) - max(LEADS.values())):
            row: dict[str, float | int] = {
                "sample_id": f"{int(year)}_{int(group.loc[idx, 'day'])}",
                "year": int(year),
                "day": int(group.loc[idx, "day"]),
                "window_size": int(window_size),
            }
            history = weather.iloc[idx - window_size + 1 : idx + 1]
            for lag in range(window_size):
                values = history.iloc[-(lag + 1)]
                for variable in FORECAST_VARIABLES:
                    row[f"{variable}_lag_{lag}"] = values[variable]
            for lead_name, lead in LEADS.items():
                future = weather.iloc[idx + lead]
                for variable in FORECAST_VARIABLES:
                    row[target_column(lead_name, variable)] = future[variable]
                    row[baseline_column(lead_name, variable)] = row[f"{variable}_lag_0"]
            rows.append(row)
    if not rows:
        raise ValueError(f"No supervised weather windows were created for W={window_size}.")
    frame = pd.DataFrame(rows)
    target_cols = [target_column(lead, variable) for lead in LEADS for variable in FORECAST_VARIABLES]
    frame = frame.dropna(subset=target_cols).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"All supervised weather windows for W={window_size} had missing target weather values.")
    feature_cols = [f"{variable}_lag_{lag}" for lag in range(window_size) for variable in FORECAST_VARIABLES]
    assert_no_forbidden_features(feature_cols)
    return frame, feature_cols


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    columns = list(columns)
    bad: list[str] = []
    for col in columns:
        normalized = normalize_name(col)
        if normalized in FORBIDDEN_COLUMNS or normalized.startswith("target_favorable") or normalized.startswith("target_h"):
            bad.append(col)
    if bad:
        raise ValueError(f"Forbidden disease/formula columns selected for weather benchmark: {bad}")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    values = np.zeros_like(denom, dtype=float)
    np.divide(2.0 * np.abs(y_pred - y_true), denom, out=values, where=denom != 0)
    return float(np.mean(values))


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def clip_prediction(variable: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if variable == "precipitation":
        return np.clip(arr, 0.0, None)
    if variable == "cloudiness":
        return np.clip(arr, 0.0, 10.0)
    if variable == RAIN_TARGET:
        return np.clip(arr, 0.0, 1.0)
    return arr


def optimize_blend_weight(
    y_val: np.ndarray,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    variable: str,
) -> float:
    grid = np.linspace(0.0, 1.0, 21)
    best_weight = 1.0
    best_score = float("inf")
    for weight in grid:
        blended = weight * model_pred + (1.0 - weight) * baseline_pred
        blended = clip_prediction(variable, blended)
        score = brier_score_loss(y_val.astype(int), blended) if variable == RAIN_TARGET else mean_absolute_error(y_val, blended)
        if score < best_score:
            best_score = float(score)
            best_weight = float(weight)
    return best_weight


def optimize_rain_threshold(y_val: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), np.round(y_score, 8)]))
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        pred = (y_score >= threshold).astype(int)
        key = (
            f1_score(y_val, pred, zero_division=0),
            precision_score(y_val, pred, zero_division=0),
            recall_score(y_val, pred, zero_division=0),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def tree_default_params(model_name: str, is_classifier: bool, iterations: int, random_state: int) -> dict[str, Any]:
    if model_name == "catboost_hybrid":
        params = {
            "iterations": iterations,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 6.0,
            "loss_function": "Logloss" if is_classifier else "RMSE",
            "random_seed": random_state,
            "allow_writing_files": False,
            "verbose": False,
        }
        if is_classifier:
            params["auto_class_weights"] = "Balanced"
        if not is_classifier:
            params["eval_metric"] = "RMSE"
        return params
    params = {
        "n_estimators": iterations,
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_samples": 10,
        "reg_lambda": 1.0,
        "objective": "binary" if is_classifier else "regression",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if is_classifier:
        params["class_weight"] = "balanced"
    return params


def suggest_tree_params(trial: Any, model_name: str, is_classifier: bool, iterations: int) -> dict[str, Any]:
    if model_name == "catboost_hybrid":
        params = tree_default_params(model_name, is_classifier, iterations, RANDOM_STATE)
        params.update(
            {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
            }
        )
        return params
    params = tree_default_params(model_name, is_classifier, iterations, RANDOM_STATE)
    params.update(
        {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    )
    return params


def build_tree_estimator(model_name: str, params: dict[str, Any], is_classifier: bool) -> Any:
    if model_name == "catboost_hybrid":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise ImportError("catboost_hybrid requires catboost. Install requirements.txt on Kaggle.") from exc
        return CatBoostClassifier(**params) if is_classifier else CatBoostRegressor(**params)
    if model_name == "lightgbm_hybrid":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm_hybrid requires lightgbm. Install requirements.txt on Kaggle.") from exc
        return LGBMClassifier(**params) if is_classifier else LGBMRegressor(**params)
    raise ValueError(model_name)


def fit_tree_estimator(
    model_name: str,
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int,
) -> Any:
    if model_name == "catboost_hybrid":
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if len(x_val):
            fit_kwargs["eval_set"] = (x_val, y_val)
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        estimator.fit(x_train, y_train, **fit_kwargs)
        return estimator
    if model_name == "lightgbm_hybrid":
        fit_kwargs: dict[str, Any] = {}
        if len(x_val):
            try:
                import lightgbm as lgb

                fit_kwargs["eval_set"] = [(x_val, y_val)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
            except Exception:
                fit_kwargs["eval_set"] = [(x_val, y_val)]
        estimator.fit(x_train, y_train, **fit_kwargs)
        return estimator
    raise ValueError(model_name)


def predict_tree(estimator: Any, x: pd.DataFrame, is_classifier: bool) -> np.ndarray:
    if is_classifier:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(x)
            return np.asarray(proba)[:, 1].astype(float)
        return np.asarray(estimator.predict(x), dtype=float)
    return np.asarray(estimator.predict(x), dtype=float)


def tune_tree_params(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    variable: str,
    trials: int,
    iterations: int,
    early_stopping_rounds: int,
) -> dict[str, Any]:
    is_classifier = variable == RAIN_TARGET
    if trials <= 0 or optuna is None:
        return tree_default_params(model_name, is_classifier, iterations, RANDOM_STATE)

    def objective(trial: Any) -> float:
        params = suggest_tree_params(trial, model_name, is_classifier, iterations)
        estimator = build_tree_estimator(model_name, params, is_classifier)
        fitted = fit_tree_estimator(model_name, estimator, x_train, y_train, x_val, y_val, early_stopping_rounds)
        pred = clip_prediction(variable, predict_tree(fitted, x_val, is_classifier))
        if is_classifier:
            return float(brier_score_loss(y_val.astype(int), pred))
        return float(mean_absolute_error(y_val, pred))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    params = tree_default_params(model_name, is_classifier, iterations, RANDOM_STATE)
    params.update(dict(study.best_params))
    return params


def train_tree_hybrid(
    model_name: str,
    samples: pd.DataFrame,
    feature_cols: list[str],
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    train = samples[samples["year"].isin(train_years)].copy()
    val = samples[samples["year"].isin(val_years)].copy()
    test = samples[samples["year"].isin(test_years)].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation, and test sample splits must be non-empty.")

    x_train_raw = train[feature_cols]
    x_val_raw = val[feature_cols]
    x_test_raw = test[feature_cols]
    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(imputer.fit_transform(x_train_raw), columns=feature_cols)
    x_val = pd.DataFrame(imputer.transform(x_val_raw), columns=feature_cols)
    x_test = pd.DataFrame(imputer.transform(x_test_raw), columns=feature_cols)
    prediction_rows = test[["sample_id", "year", "day", "window_size"]].reset_index(drop=True).copy()
    metrics: list[dict[str, Any]] = []
    heads: dict[str, FittedHead] = {}
    params_by_head: dict[str, Any] = {}

    for lead_name in LEADS:
        for variable in FORECAST_VARIABLES:
            y_col = target_column(lead_name, variable)
            base_col = baseline_column(lead_name, variable)
            is_classifier = variable == RAIN_TARGET
            y_train = train[y_col].astype(int if is_classifier else float).reset_index(drop=True)
            y_val = val[y_col].astype(int if is_classifier else float).reset_index(drop=True)
            y_test = test[y_col].astype(int if is_classifier else float).reset_index(drop=True)
            if is_classifier and y_train.nunique() < 2:
                probability = float(y_train.iloc[0]) if len(y_train) else 0.0
                params = {"constant_probability": probability, "reason": "single-class training split"}
                estimator = ConstantProbabilityModel(probability)
                val_model_pred = np.full(len(x_val), probability, dtype=float)
                test_model_pred = np.full(len(x_test), probability, dtype=float)
            else:
                params = tune_tree_params(
                    model_name,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    variable,
                    args.tune_trials,
                    args.iterations,
                    args.early_stopping_rounds,
                )
                estimator = build_tree_estimator(model_name, params, is_classifier)
                estimator = fit_tree_estimator(model_name, estimator, x_train, y_train, x_val, y_val, args.early_stopping_rounds)
                val_model_pred = clip_prediction(variable, predict_tree(estimator, x_val, is_classifier))
                test_model_pred = clip_prediction(variable, predict_tree(estimator, x_test, is_classifier))
            val_baseline = clip_prediction(variable, val[base_col].to_numpy(dtype=float))
            test_baseline = clip_prediction(variable, test[base_col].to_numpy(dtype=float))
            blend_weight = optimize_blend_weight(y_val.to_numpy(), val_model_pred, val_baseline, variable)
            val_blended = clip_prediction(variable, blend_weight * val_model_pred + (1.0 - blend_weight) * val_baseline)
            test_blended = clip_prediction(variable, blend_weight * test_model_pred + (1.0 - blend_weight) * test_baseline)
            threshold = optimize_rain_threshold(y_val.to_numpy(dtype=int), val_blended) if is_classifier else None
            pred_col = f"{lead_name}_{variable}_pred"
            true_col = f"{lead_name}_{variable}_true"
            prediction_rows[true_col] = y_test.to_numpy()
            if is_classifier:
                prediction_rows[f"{lead_name}_{variable}_score"] = test_blended
                prediction_rows[pred_col] = (test_blended >= float(threshold)).astype(int)
            else:
                prediction_rows[pred_col] = test_blended
            metrics.extend(
                evaluate_target(
                    model_name,
                    int(samples["window_size"].iloc[0]),
                    lead_name,
                    variable,
                    y_test.to_numpy(),
                    test_blended,
                    threshold,
                    blend_weight,
                )
            )
            head_name = f"{lead_name}_{variable}"
            heads[head_name] = FittedHead(estimator, imputer, blend_weight, threshold)
            params_by_head[head_name] = params
    config = {
        "model": model_name,
        "window_size": int(samples["window_size"].iloc[0]),
        "feature_columns": feature_cols,
        "target_columns": [target_column(lead, var) for lead in LEADS for var in FORECAST_VARIABLES],
        "forbidden_columns_not_used": sorted(FORBIDDEN_COLUMNS),
        "hybrid": "direct CatBoost/LightGBM forecasts blended with raw-weather persistence using validation MAE/Brier loss",
        "params_by_head": params_by_head,
        "trained_heads": list(heads),
    }
    return prediction_rows, metrics, config | {"_heads": heads}


def evaluate_target(
    model: str,
    window_size: int,
    lead_name: str,
    variable: str,
    y_true: np.ndarray,
    y_pred_or_score: np.ndarray,
    threshold: float | None,
    blend_weight: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if variable == RAIN_TARGET:
        score = clip_prediction(variable, y_pred_or_score)
        pred = (score >= float(threshold if threshold is not None else 0.5)).astype(int)
        rows.append(
            {
                "model": model,
                "window_size": window_size,
                "lead": lead_name,
                "variable": variable,
                "metric_family": "classification",
                "accuracy": float(accuracy_score(y_true.astype(int), pred)),
                "f1": float(f1_score(y_true.astype(int), pred, zero_division=0)),
                "precision": float(precision_score(y_true.astype(int), pred, zero_division=0)),
                "recall": float(recall_score(y_true.astype(int), pred, zero_division=0)),
                "roc_auc": safe_roc_auc(y_true.astype(int), score),
                "brier": float(brier_score_loss(y_true.astype(int), score)),
                "threshold": threshold,
                "blend_weight": blend_weight,
            }
        )
    else:
        pred = clip_prediction(variable, y_pred_or_score)
        rows.append(
            {
                "model": model,
                "window_size": window_size,
                "lead": lead_name,
                "variable": variable,
                "metric_family": "regression",
                "mae": float(mean_absolute_error(y_true, pred)),
                "rmse": safe_rmse(y_true, pred),
                "smape": smape(y_true, pred),
                "r2": safe_r2(y_true, pred),
                "blend_weight": blend_weight,
            }
        )
    return rows


def run_chronos2(
    samples: pd.DataFrame,
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "chronos2 requires chronos-forecasting>=2.0. Install requirements.txt on Kaggle, "
            "or run only --models catboost_hybrid lightgbm_hybrid."
        ) from exc

    test = samples[samples["year"].isin(test_years)].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError("Chronos-2 test sample split is empty.")
    window_size = int(test["window_size"].iloc[0])
    pipeline = Chronos2Pipeline.from_pretrained(args.chronos_model_id, device_map=args.chronos_device_map)
    prediction_rows = test[["sample_id", "year", "day", "window_size"]].copy()
    metrics: list[dict[str, Any]] = []
    all_preds: dict[tuple[str, str], list[float]] = {(lead, var): [] for lead in LEADS for var in FORECAST_VARIABLES}

    for start in range(0, len(test), args.chronos_batch_size):
        batch = test.iloc[start : start + args.chronos_batch_size].reset_index(drop=True)
        context_rows: list[dict[str, Any]] = []
        for row_idx, row in batch.iterrows():
            item_id = str(row["sample_id"])
            for ts in range(window_size):
                context_row = {"id": item_id, "timestamp": ts}
                lag = window_size - 1 - ts
                for variable in FORECAST_VARIABLES:
                    context_row[variable] = row[f"{variable}_lag_{lag}"]
                context_rows.append(context_row)
        context_df = pd.DataFrame(context_rows)
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=2,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target=FORECAST_VARIABLES,
        )
        extracted = extract_chronos_predictions(pred_df, batch["sample_id"].astype(str).tolist())
        for sample_id in batch["sample_id"].astype(str):
            for lead_name, lead in LEADS.items():
                for variable in FORECAST_VARIABLES:
                    all_preds[(lead_name, variable)].append(float(extracted[sample_id][lead - 1][variable]))

    for lead_name in LEADS:
        for variable in FORECAST_VARIABLES:
            y_col = target_column(lead_name, variable)
            pred = clip_prediction(variable, np.asarray(all_preds[(lead_name, variable)], dtype=float))
            true = test[y_col].to_numpy(dtype=int if variable == RAIN_TARGET else float)
            prediction_rows[f"{lead_name}_{variable}_true"] = true
            if variable == RAIN_TARGET:
                score = pred
                threshold = 0.5
                prediction_rows[f"{lead_name}_{variable}_score"] = score
                prediction_rows[f"{lead_name}_{variable}_pred"] = (score >= threshold).astype(int)
                metrics.extend(evaluate_target("chronos2", window_size, lead_name, variable, true, score, threshold, None))
            else:
                prediction_rows[f"{lead_name}_{variable}_pred"] = pred
                metrics.extend(evaluate_target("chronos2", window_size, lead_name, variable, true, pred, None, None))

    config = {
        "model": "chronos2",
        "window_size": window_size,
        "chronos_model_id": args.chronos_model_id,
        "chronos_device_map": args.chronos_device_map,
        "chronos_batch_size": args.chronos_batch_size,
        "mode": "zero-shot Chronos-2 inference on each raw weather history window",
        "train_years_not_fit": train_years,
        "validation_years_not_fit": val_years,
    }
    return prediction_rows, metrics, config


def extract_chronos_predictions(pred_df: pd.DataFrame, sample_ids: list[str]) -> dict[str, list[dict[str, float]]]:
    """Extract median Chronos-2 predictions from common predict_df shapes."""
    frame = pred_df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = ["_".join(str(part) for part in col if str(part)) for col in frame.columns]
    id_col = "id" if "id" in frame.columns else "item_id" if "item_id" in frame.columns else frame.columns[0]
    frame[id_col] = frame[id_col].astype(str)
    result: dict[str, list[dict[str, float]]] = {}
    for sample_id in sample_ids:
        group = frame[frame[id_col] == sample_id].copy()
        if group.empty:
            raise ValueError(f"Chronos-2 prediction output missing sample_id={sample_id}")
        if "timestamp" in group.columns:
            group = group.sort_values("timestamp")
        group = group.tail(2).reset_index(drop=True)
        leads: list[dict[str, float]] = []
        for _, row in group.iterrows():
            lead_values: dict[str, float] = {}
            for variable in FORECAST_VARIABLES:
                candidates = [
                    variable,
                    f"{variable}_0.5",
                    f"{variable}_q0.5",
                    f"{variable}_median",
                    f"{variable}_mean",
                    f"0.5_{variable}",
                    f"median_{variable}",
                ]
                found = next((col for col in candidates if col in group.columns), None)
                if found is None:
                    variable_matches = [col for col in group.columns if str(col).startswith(f"{variable}_")]
                    found = variable_matches[0] if variable_matches else None
                if found is None:
                    raise ValueError(
                        "Could not parse Chronos-2 predict_df output. "
                        f"Missing median column for {variable}; columns={list(group.columns)}"
                    )
                lead_values[variable] = float(row[found])
            leads.append(lead_values)
        result[sample_id] = leads
    return result


def aggregate_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(metrics)
    regression = frame[frame["metric_family"] == "regression"]
    classification = frame[frame["metric_family"] == "classification"]
    summary: dict[str, Any] = {
        "mean_mae": float(regression["mae"].mean()) if not regression.empty else float("nan"),
        "mean_rmse": float(regression["rmse"].mean()) if not regression.empty else float("nan"),
        "mean_smape": float(regression["smape"].mean()) if not regression.empty else float("nan"),
        "mean_r2": float(regression["r2"].mean()) if not regression.empty else float("nan"),
        "rain_accuracy": float(classification["accuracy"].mean()) if not classification.empty else float("nan"),
        "rain_f1": float(classification["f1"].mean()) if not classification.empty else float("nan"),
        "rain_brier": float(classification["brier"].mean()) if not classification.empty else float("nan"),
    }
    summary["weather_score_lower_is_better"] = float(
        np.nanmean(
            [
                summary["mean_smape"],
                summary["rain_brier"],
                1.0 - summary["rain_f1"] if not math.isnan(summary["rain_f1"]) else float("nan"),
            ]
        )
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def safe_run_dir(output_dir: Path, model: str, window_size: int) -> Path:
    base = output_dir / f"{model}_w{window_size}"
    path = base
    suffix = 2
    while path.exists():
        path = output_dir / f"{base.name}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def zip_dir(run_dir: Path, output_dir: Path) -> Path:
    zips_dir = output_dir / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zips_dir / f"{run_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir.parent))
    return zip_path


def maybe_upload(zip_path: Path, output_dir: Path, upload: bool, dataset_slug: str | None) -> None:
    if not upload:
        warnings.warn(
            f"Zip saved locally at {zip_path}. Kaggle /working files are ephemeral; "
            "use --upload-dataset --dataset-slug username/dataset-name to stage a Dataset copy.",
            RuntimeWarning,
        )
        return
    if not dataset_slug:
        raise ValueError("--dataset-slug is required when --upload-dataset is set.")
    persistent_dir = output_dir / "persistent_dataset"
    persistent_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, persistent_dir / zip_path.name)
    metadata = {"id": dataset_slug, "title": dataset_slug.split("/", 1)[-1]}
    (persistent_dir / "dataset-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def save_model_artifacts(run_dir: Path, heads: dict[str, FittedHead] | None) -> None:
    if heads is None:
        return
    if joblib is None:
        (run_dir / "skipped_model_artifacts.txt").write_text("joblib is not installed; fitted tree heads were not serialized.\n", encoding="utf-8")
        return
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for name, head in heads.items():
        joblib.dump(head, models_dir / f"{name}.joblib")


def save_run(
    output_dir: Path,
    model_name: str,
    window_size: int,
    predictions: pd.DataFrame | None,
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
    splits: dict[str, list[int]],
    skipped_reason: str | None = None,
    upload: bool = False,
    dataset_slug: str | None = None,
) -> ModelRun:
    run_dir = safe_run_dir(output_dir, model_name, window_size)
    config_for_disk = dict(config)
    heads = config_for_disk.pop("_heads", None)
    if predictions is not None:
        predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_csv(run_dir / "metrics_weather.csv", metrics)
    write_json(run_dir / "summary_weather.json", aggregate_summary(metrics) if metrics else {})
    write_json(run_dir / "config_used.json", config_for_disk)
    write_json(run_dir / "train_val_test_years.json", splits)
    if skipped_reason:
        (run_dir / "skipped_model.txt").write_text(skipped_reason + "\n", encoding="utf-8")
    save_model_artifacts(run_dir, heads)
    zip_path = zip_dir(run_dir, output_dir)
    maybe_upload(zip_path, output_dir, upload, dataset_slug)
    return ModelRun(model_name, window_size, run_dir, zip_path, metrics)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = validate_models(args.models)
    windows = validate_window_sizes(args.window_sizes)
    data_path = find_data_file(args.data)
    weather = load_weather_dataset(data_path)
    assert_no_forbidden_features(FORECAST_VARIABLES)
    train_years, val_years, test_years = split_years(weather, args.val_year_count, args.test_year_count)
    splits = {"train": train_years, "validation": val_years, "test": test_years}
    print(f"Weather data: {data_path} rows={len(weather)} years={weather['year'].nunique()}")
    print(f"Train years: {train_years}")
    print(f"Validation years: {val_years}")
    print(f"Test years: {test_years}")
    if args.tune_trials > 0 and optuna is None:
        warnings.warn("Optuna is not installed; running default boosting parameters.", RuntimeWarning)

    all_summary_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    for window_size in windows:
        samples, feature_cols = make_weather_windows(weather, window_size)
        for model_name in models:
            start_time = time.time()
            try:
                if model_name in {"catboost_hybrid", "lightgbm_hybrid"}:
                    predictions, metrics, config = train_tree_hybrid(model_name, samples, feature_cols, train_years, val_years, test_years, args)
                elif model_name == "chronos2":
                    predictions, metrics, config = run_chronos2(samples, train_years, val_years, test_years, args)
                else:
                    raise ValueError(model_name)
                config = config | {
                    "data_path": str(data_path),
                    "forecast_variables": FORECAST_VARIABLES,
                    "leads": LEADS,
                    "input_contract": f"raw weather history only, W={window_size} days",
                    "disease_classifier_trained": False,
                    "target_favorable_predicted": False,
                }
                run = save_run(
                    args.output_dir,
                    model_name,
                    window_size,
                    predictions,
                    metrics,
                    config,
                    splits,
                    upload=args.upload_dataset,
                    dataset_slug=args.dataset_slug,
                )
                summary = aggregate_summary(metrics)
                summary_row = {
                    "model": model_name,
                    "window_size": window_size,
                    "run_dir": str(run.run_dir),
                    "zip_path": str(run.zip_path),
                    "seconds": round(time.time() - start_time, 3),
                    **summary,
                }
                all_summary_rows.append(summary_row)
                all_metrics.extend(metrics)
                print(
                    f"DONE {model_name} W={window_size} "
                    f"score={summary['weather_score_lower_is_better']:.4f} -> {run.run_dir}"
                )
            except Exception as exc:
                reason = f"{model_name} W={window_size} skipped/failed: {type(exc).__name__}: {exc}"
                warnings.warn(reason, RuntimeWarning)
                run = save_run(
                    args.output_dir,
                    model_name,
                    window_size,
                    None,
                    [],
                    {
                        "model": model_name,
                        "window_size": window_size,
                        "data_path": str(data_path),
                        "error": reason,
                        "disease_classifier_trained": False,
                        "target_favorable_predicted": False,
                    },
                    splits,
                    skipped_reason=reason,
                    upload=args.upload_dataset,
                    dataset_slug=args.dataset_slug,
                )
                all_summary_rows.append(
                    {
                        "model": model_name,
                        "window_size": window_size,
                        "run_dir": str(run.run_dir),
                        "zip_path": str(run.zip_path),
                        "error": reason,
                    }
                )
                print(f"SKIP {reason}")

    summary_df = pd.DataFrame(all_summary_rows)
    if "weather_score_lower_is_better" in summary_df.columns:
        summary_df = summary_df.sort_values(["weather_score_lower_is_better", "model", "window_size"], na_position="last")
    summary_df.to_csv(args.output_dir / "summary_weather.csv", index=False)
    write_csv(args.output_dir / "metrics_weather_all.csv", all_metrics)
    write_json(
        args.output_dir / "weather_benchmark_contract.json",
        {
            "input": "raw weather history for W days where W is 7 or 11",
            "outputs": {"lead_1": "t+1", "lead_2": "t+2"},
            "forecast_variables": FORECAST_VARIABLES,
            "models": SUPPORTED_MODELS,
            "forbidden_columns_not_used": sorted(FORBIDDEN_COLUMNS),
            "disease_classifier_trained": False,
            "target_favorable_predicted": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
