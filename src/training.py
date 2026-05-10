from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .artifacts import copy_zip_to_persistent, save_experiment_artifacts, upload_dataset_version
from .data import create_horizon_targets, load_excel_dataset, split_years
from .evaluation import evaluate_binary
from .features import add_engineered_features, make_xy
from .models import fit_predict_sklearn, predict_blitecast, predict_sarima

DEFAULT_OUTPUT_DIR = "/kaggle/working/outputs"


def run_experiments(
    excel_path: str | Path = "Golitcino72-17d2_CLEAN.xlsx",
    models: Iterable[str] = ("logreg",),
    horizons: Iterable[int] = (2,),
    window_sizes: Iterable[int] = (7,),
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int | None = None,
    tune_trials: int = 1,
    upload_dataset: bool = False,
    dataset_slug: str | None = None,
    no_y: bool = False,
    random_state: int = 42,
) -> list[dict]:
    if upload_dataset and not dataset_slug:
        raise ValueError("--dataset-slug is required when --upload-dataset is used")
    model_names = list(models)
    if model_names == ["all"]:
        model_names = ["logreg", "svm", "rf", "gru", "tft", "blitecast", "xgboost", "catboost", "lightgbm", "arima", "sarima"]
    horizons = [int(h) for h in horizons]
    window_sizes = [int(w) for w in window_sizes]
    data = create_horizon_targets(load_excel_dataset(excel_path), horizons)
    splits = split_years(data)
    output_dir = Path(output_dir)
    results: list[dict] = []

    for horizon in horizons:
        target_col = f"target_h{horizon}"
        feature_frame, features = add_engineered_features(data, window_sizes, no_y=no_y)
        train_df = feature_frame[feature_frame["year"].isin(splits["train"])]
        val_df = feature_frame[feature_frame["year"].isin(splits["val"])]
        test_df = feature_frame[feature_frame["year"].isin(splits["test"])]
        X_train, y_train, _ = make_xy(train_df, features, target_col)
        X_val, y_val, _ = make_xy(val_df, features, target_col)
        X_test, y_test, meta_test = make_xy(test_df, features, target_col)
        for model_name in model_names:
            if model_name in {"logreg", "rf"}:
                fitted = fit_predict_sklearn(
                    model_name,
                    X_train,
                    y_train,
                    X_test,
                    X_val=X_val,
                    y_val=y_val,
                    random_state=random_state,
                )
            elif model_name == "blitecast":
                aligned = test_df.dropna(subset=[target_col]).copy()
                fitted = predict_blitecast(aligned)
            elif model_name == "sarima":
                fitted = predict_sarima(y_train, len(y_test))
            else:
                raise ValueError(f"Unsupported model for this worker slice: {model_name}")
            metrics, report = evaluate_binary(y_test, fitted.y_pred_binary)
            predictions = meta_test.reset_index(drop=True).copy()
            predictions["y_true"] = list(y_test)
            predictions["y_pred_binary"] = fitted.y_pred_binary
            if fitted.y_score is not None:
                predictions["y_score"] = fitted.y_score
            config = {
                "excel_path": str(excel_path),
                "model": model_name,
                "horizon": horizon,
                "window_sizes": window_sizes,
                "top_n": top_n,
                "tune_trials": tune_trials,
                "no_y": no_y,
                "primary_metric": "f1",
            }
            if model_name == "logreg":
                config["threshold"] = fitted.threshold
                config["threshold_tuning"] = "validation_precision_floor_then_recall"
                config["validation_metrics"] = fitted.validation_metrics or {}
                config["validation_threshold_objective"] = fitted.threshold_objective
            run_dir, zip_path = save_experiment_artifacts(output_dir, model_name, metrics, predictions, report, config, splits)
            if upload_dataset:
                persistent_dir = Path(output_dir) / "persistent_dataset"
                copy_zip_to_persistent(zip_path, persistent_dir, dataset_slug or "")
                upload_dataset_version(persistent_dir)
            results.append({
                "model": model_name,
                "horizon": horizon,
                **metrics,
                "threshold": fitted.threshold,
                "run_dir": str(run_dir),
                "zip_path": str(zip_path),
            })
    results.sort(key=lambda row: (row["model"], -row["f1"]))
    if top_n is not None:
        limit = int(top_n)
        if limit <= 0:
            return []
        kept: list[dict] = []
        per_model_counts: dict[str, int] = {}
        for row in sorted(results, key=lambda item: item["f1"], reverse=True):
            model = str(row["model"])
            count = per_model_counts.get(model, 0)
            if count < limit:
                kept.append(row)
                per_model_counts[model] = count + 1
        return sorted(kept, key=lambda row: (row["model"], -row["f1"]))
    return results
