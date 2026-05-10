from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "t_min",
    "t_max",
    "is_rain",
    "precipitation",
    "t_avg",
    "cloudiness",
    "y1",
    "y2",
    "y2_y1",
    "y3",
    "y4",
    "precipitation_t_gt_10",
    "t_gt_10",
]
ZERO_FILL_COLUMNS = ["precipitation", "precipitation_t_gt_10", "is_rain"]
Y_FEATURE_COLUMNS = {"y1", "y2", "y2_y1", "y3", "y4", "fe_y4_cloud_interaction"}


def is_y_feature(column: str) -> bool:
    if column in Y_FEATURE_COLUMNS:
        return True
    for source in ("y1", "y2", "y2_y1", "y3", "y4"):
        if column.startswith((f"{source}_lag_", f"{source}_roll", f"{source}_trend_")):
            return True
    return False


def add_engineered_features(df: pd.DataFrame, window_sizes: Iterable[int], no_y: bool = False) -> tuple[pd.DataFrame, list[str]]:
    result = df.sort_values(["year", "day"]).copy()
    for column in ZERO_FILL_COLUMNS:
        if column in result.columns:
            result[column] = result[column].fillna(0)
    if {"t_avg", "cloudiness"}.issubset(result.columns):
        result["fe_cloud_temp"] = result["t_avg"] * result["cloudiness"]
    if {"precipitation", "t_avg"}.issubset(result.columns):
        result["fe_cold_rain_index"] = result["precipitation"] * (20 - result["t_avg"]).clip(lower=0)

    created: list[str] = []
    source_cols = [col for col in ["t_avg", "precipitation", "cloudiness", "y4"] if col in result.columns]
    if no_y:
        source_cols = [col for col in source_cols if not is_y_feature(col)]
    for window in sorted(set(int(w) for w in window_sizes)):
        if window <= 0:
            raise ValueError("window sizes must be positive")
        for column in source_cols:
            name = f"{column}_roll{window}_mean"
            result[name] = result.groupby("year", sort=False)[column].transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )
            created.append(name)
    features = [col for col in BASE_FEATURES if col in result.columns]
    features.extend([col for col in ["fe_cloud_temp", "fe_cold_rain_index"] if col in result.columns])
    features.extend(created)
    if no_y:
        features = [col for col in features if not is_y_feature(col)]
    return result, features


def make_xy(df: pd.DataFrame, features: list[str], target_col: str):
    frame = df.dropna(subset=[target_col]).copy()
    X = frame[features].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0)
    X = X.fillna(medians).fillna(0)
    y = frame[target_col].astype(int)
    meta = frame[["year", "day"]].copy()
    return X, y, meta
