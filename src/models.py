"""Model factories transferred from the diploma notebooks.

This module keeps the mandatory model implementations in one simple place so
``train.py`` can select models by CLI names.  The tree-boosting defaults and
Optuna search spaces mirror ``potato_illness_forecasting_benchmark (2).ipynb``;
the ARIMA/SARIMA walk-forward wrapper follows the statistical-model baseline
used in the main notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluation import evaluate_binary, search_threshold

RANDOM_STATE = 42

MODEL_ALIASES = {
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "cat": "catboost",
    "catboost": "catboost",
    "arima": "arima",
    "sarima": "sarima",
    "sarimax": "sarima",
    "logreg": "logreg",
    "svm": "svm",
    "rf": "rf",
}

TRANSFERRED_MODEL_NAMES = ("xgboost", "catboost", "lightgbm", "arima", "sarima")
LOGREG_SOLVERS = ("liblinear", "lbfgs")


def normalize_model_name(model_name: str) -> str:
    """Return canonical CLI model name or fail with a clear error."""
    key = model_name.strip().lower().replace("-", "_")
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown model '{model_name}'. Supported: {sorted(MODEL_ALIASES)}")
    return MODEL_ALIASES[key]


def require_optional(import_name: str, package_name: str):
    """Import optional Kaggle dependency lazily with an actionable message."""
    try:
        module = __import__(import_name, fromlist=["*"])
    except ImportError as exc:
        raise ImportError(
            f"Model dependency '{package_name}' is not installed. "
            f"Install requirements.txt before running this model."
        ) from exc
    return module


def get_class_ratio(y: Iterable[int]) -> float:
    """Negative/positive ratio used as XGBoost scale_pos_weight."""
    y_series = pd.Series(y).astype(int)
    positives = max(1, int(y_series.sum()))
    negatives = max(1, int((1 - y_series).sum()))
    return negatives / positives


def sanitize_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Match notebook behavior: numeric float32 features, duplicate columns removed."""
    frame = pd.DataFrame(X).copy()
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.astype(np.float32)


def sanitize_xy(X: pd.DataFrame, y: Iterable[int]) -> Tuple[pd.DataFrame, pd.Series]:
    X_clean = sanitize_feature_frame(X).reset_index(drop=True)
    y_clean = pd.Series(y).reset_index(drop=True).astype(int)
    return X_clean, y_clean


def normalize_logreg_params(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(params or {})
    params.setdefault("C", 1.0)
    params.setdefault("solver", "liblinear")
    params.setdefault("penalty", "l2")
    params.setdefault("class_weight", None)
    params.setdefault("max_iter", 1000)
    params.pop("l1_ratio", None)
    return params


def suggest_logreg_params(trial) -> Dict[str, Any]:
    return normalize_logreg_params({
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "solver": trial.suggest_categorical("solver", list(LOGREG_SOLVERS)),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    })


def build_logreg(params: Optional[Dict[str, Any]] = None):
    params = normalize_logreg_params(params)
    clf_kwargs = {
        "C": params["C"],
        "solver": params["solver"],
        "penalty": params["penalty"],
        "class_weight": params["class_weight"],
        "random_state": RANDOM_STATE,
        "max_iter": params["max_iter"],
    }
    if "l1_ratio" in params:
        clf_kwargs["l1_ratio"] = params["l1_ratio"]
    clf = LogisticRegression(
        **clf_kwargs,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_svm(params: Optional[Dict[str, Any]] = None):
    params = params or {}
    clf = SVC(
        C=params.get("C", 1.0),
        kernel=params.get("kernel", "rbf"),
        gamma=params.get("gamma", "scale"),
        class_weight=params.get("class_weight"),
        probability=True,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_rf(params: Optional[Dict[str, Any]] = None):
    params = params or {}
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth"),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        class_weight=params.get("class_weight"),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgboost(params: Optional[Dict[str, Any]] = None, scale_pos_weight: float = 1.0):
    params = params or {}
    xgb = require_optional("xgboost", "xgboost")
    return xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        min_child_weight=params.get("min_child_weight", 1.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=params.get("scale_pos_weight", scale_pos_weight),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def build_lightgbm(params: Optional[Dict[str, Any]] = None):
    params = params or {}
    lgb = require_optional("lightgbm", "lightgbm")
    return lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 300),
        num_leaves=params.get("num_leaves", 31),
        max_depth=params.get("max_depth", -1),
        learning_rate=params.get("learning_rate", 0.05),
        min_child_samples=params.get("min_child_samples", 20),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_lambda=params.get("reg_lambda", 1.0),
        class_weight=params.get("class_weight"),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_catboost(params: Optional[Dict[str, Any]] = None):
    params = params or {}
    catboost = require_optional("catboost", "catboost")
    return catboost.CatBoostClassifier(
        iterations=params.get("iterations", 300),
        depth=params.get("depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
        random_strength=params.get("random_strength", 1.0),
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
    )


@dataclass
class ARIMABinaryForecaster:
    """Simple ARIMA/SARIMA probability-style forecaster for binary targets.

    The model treats the binary target as a continuous 0/1 series, forecasts the
    next values, clips forecasts into [0, 1], and thresholds them for binary
    classification.  This matches the notebook's baseline framing for
    statistical models without pretending ROC/PR scores are available when the
    fitted model fails to produce meaningful numeric forecasts.
    """

    order: Tuple[int, int, int] = (1, 0, 0)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    threshold: float = 0.5
    _result: Any = None
    _exog_columns: Optional[list[str]] = None

    def fit(self, y: Iterable[int], exog: Optional[pd.DataFrame] = None):
        sm = require_optional("statsmodels.tsa.statespace.sarimax", "statsmodels")
        y_series = pd.Series(y).astype(float).reset_index(drop=True)
        exog_values = None
        if exog is not None:
            exog_frame = sanitize_feature_frame(exog).reset_index(drop=True)
            self._exog_columns = list(exog_frame.columns)
            exog_values = exog_frame.values
        model = sm.SARIMAX(
            y_series.values,
            exog=exog_values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        return self

    def predict_score(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("ARIMA/SARIMA model must be fitted before prediction.")
        exog_values = None
        if exog is not None:
            exog_values = sanitize_feature_frame(exog).values
        forecast = self._result.forecast(steps=steps, exog=exog_values)
        return np.clip(np.asarray(forecast, dtype=float).reshape(-1), 0.0, 1.0)

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        return (self.predict_score(steps=steps, exog=exog) >= self.threshold).astype(int)


def build_arima(params: Optional[Dict[str, Any]] = None) -> ARIMABinaryForecaster:
    params = params or {}
    return ARIMABinaryForecaster(
        order=tuple(params.get("order", (1, 0, 0))),
        seasonal_order=(0, 0, 0, 0),
        threshold=params.get("threshold", 0.5),
    )


def build_sarima(params: Optional[Dict[str, Any]] = None) -> ARIMABinaryForecaster:
    params = params or {}
    return ARIMABinaryForecaster(
        order=tuple(params.get("order", (1, 0, 0))),
        seasonal_order=tuple(params.get("seasonal_order", (1, 0, 0, 7))),
        threshold=params.get("threshold", 0.5),
    )


def build_model(model_name: str, params: Optional[Dict[str, Any]] = None, y_train: Optional[Iterable[int]] = None):
    """Build a model by CLI name."""
    name = normalize_model_name(model_name)
    params = params or {}
    if name == "logreg":
        return build_logreg(params)
    if name == "svm":
        return build_svm(params)
    if name == "rf":
        return build_rf(params)
    if name == "xgboost":
        ratio = get_class_ratio(y_train) if y_train is not None else 1.0
        return build_xgboost(params, scale_pos_weight=ratio)
    if name == "lightgbm":
        return build_lightgbm(params)
    if name == "catboost":
        return build_catboost(params)
    if name == "arima":
        return build_arima(params)
    if name == "sarima":
        return build_sarima(params)
    raise ValueError(f"Unsupported model '{model_name}'")


def fit_estimator(model_name: str, model, X_train, y_train, X_val=None, y_val=None):
    """Fit sklearn-compatible boosting models with notebook-style eval sets."""
    name = normalize_model_name(model_name)
    if name in {"arima", "sarima"}:
        return model.fit(y_train, exog=X_train)

    X_train, y_train = sanitize_xy(X_train, y_train)
    if X_val is not None and y_val is not None:
        X_val, y_val = sanitize_xy(X_val, y_val)

    if name == "xgboost" and X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "lightgbm" and X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    elif name == "catboost" and X_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def predict_scores(model_name: str, model, X) -> Optional[np.ndarray]:
    """Return positive-class scores when a model naturally exposes them."""
    name = normalize_model_name(model_name)
    if name in {"arima", "sarima"}:
        return model.predict_score(steps=len(X), exog=X)
    X_clean = sanitize_feature_frame(X)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X_clean))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_clean), dtype=float)
    return None


def predict_binary(model_name: str, model, X, threshold: float = 0.5) -> np.ndarray:
    """Return mandatory binary predictions for the common classification interface."""
    scores = predict_scores(model_name, model, X)
    if scores is not None:
        return (scores >= threshold).astype(int)
    return np.asarray(model.predict(sanitize_feature_frame(X))).astype(int)


def suggest_params(trial, model_name: str, y_train: Iterable[int]) -> Dict[str, Any]:
    """Optuna search spaces transferred from the notebooks for supported models."""
    name = normalize_model_name(model_name)
    if name == "logreg":
        return suggest_logreg_params(trial)
    ratio = get_class_ratio(y_train)
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", max(0.5, ratio * 0.5), ratio * 2.0),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, step=8),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 600, step=50),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
        }
    if name in {"arima", "sarima"}:
        order = trial.suggest_categorical("order", [(1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 0, 1)])
        params: Dict[str, Any] = {"order": order}
        if name == "sarima":
            params["seasonal_order"] = trial.suggest_categorical(
                "seasonal_order",
                [(0, 0, 0, 0), (1, 0, 0, 7), (1, 0, 1, 7)],
            )
        return params
    raise ValueError(f"No transferred Optuna search space for model '{model_name}'")

# Compatibility helpers for the simple worker-2 training orchestrator.
@dataclass
class ModelResult:
    model: Any | None
    y_score: Optional[np.ndarray]
    y_pred_binary: np.ndarray


def fit_predict_sklearn(model_name: str, X_train, y_train, X_test, random_state: int = RANDOM_STATE) -> ModelResult:
    model = build_model(model_name, y_train=y_train)
    fit_estimator(model_name, model, X_train, y_train)
    scores = predict_scores(model_name, model, X_test)
    y_pred = predict_binary(model_name, model, X_test)
    return ModelResult(model=model, y_score=scores, y_pred_binary=y_pred)


def predict_blitecast(frame: pd.DataFrame) -> ModelResult:
    precipitation = frame.get("precipitation", pd.Series(0, index=frame.index)).fillna(0)
    t_avg = frame.get("t_avg", pd.Series(0, index=frame.index)).fillna(0)
    cloudiness = frame.get("cloudiness", pd.Series(0, index=frame.index)).fillna(0)
    score = (
        0.45 * (precipitation > 0).astype(float)
        + 0.30 * t_avg.between(12, 20).astype(float)
        + 0.25 * (cloudiness >= 6).astype(float)
    ).to_numpy()
    return ModelResult(model=None, y_score=score, y_pred_binary=(score >= 0.5).astype(int))


def predict_sarima(train_y: pd.Series, test_len: int) -> ModelResult:
    model = build_sarima()
    model.fit(train_y)
    score = model.predict_score(test_len)
    return ModelResult(model=model, y_score=score, y_pred_binary=(score >= 0.5).astype(int))

@dataclass
class PredictionResult:
    """Common prediction payload used by training.py."""

    y_pred_binary: np.ndarray
    y_score: Optional[np.ndarray] = None
    threshold: float = 0.5
    validation_metrics: Optional[dict[str, float]] = None
    threshold_objective: Optional[float] = None


def fit_predict_sklearn(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: Iterable[int],
    X_test: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[Iterable[int]] = None,
    random_state: int = RANDOM_STATE,
) -> PredictionResult:
    """Backward-compatible helper for the simple training pipeline."""
    name = normalize_model_name(model_name)
    params: Dict[str, Any] = {}
    if name == "rf":
        params = {"n_estimators": 200, "class_weight": "balanced"}
    model = build_model(name, params, y_train)
    fitted = fit_estimator(name, model, X_train, y_train)
    threshold = 0.5
    validation_metrics: Optional[dict[str, float]] = None
    threshold_objective: Optional[float] = None
    if name == "logreg" and X_val is not None and y_val is not None and len(X_val) > 0:
        val_score = predict_scores(name, fitted, X_val)
        if val_score is not None:
            threshold, threshold_info = search_threshold(y_val, val_score)
            threshold_objective = threshold_info["objective"]
            val_pred = (val_score >= threshold).astype(int)
            validation_metrics, _ = evaluate_binary(y_val, val_pred)
    y_score = predict_scores(name, fitted, X_test)
    if y_score is not None:
        y_pred_binary = (y_score >= threshold).astype(int)
    else:
        y_pred_binary = predict_binary(name, fitted, X_test, threshold=threshold)
    return PredictionResult(
        y_pred_binary=y_pred_binary,
        y_score=y_score,
        threshold=threshold,
        validation_metrics=validation_metrics,
        threshold_objective=threshold_objective,
    )


def predict_blitecast(frame: pd.DataFrame) -> PredictionResult:
    """Simple notebook baseline: favorable if recent weather signals are high."""
    if frame.empty:
        return PredictionResult(y_pred_binary=np.array([], dtype=int), y_score=np.array([], dtype=float))
    candidates = [col for col in ["target_favorable", "is_rain", "precipitation_t_gt_10", "t_gt_10"] if col in frame]
    if not candidates:
        scores = np.zeros(len(frame), dtype=float)
    else:
        numeric = frame[candidates].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        max_values = numeric.max().replace(0, 1)
        scores = (numeric / max_values).mean(axis=1).clip(0, 1).to_numpy(dtype=float)
    return PredictionResult(y_pred_binary=(scores >= 0.5).astype(int), y_score=scores)


def predict_sarima(y_train: Iterable[int], steps: int) -> PredictionResult:
    """Compatibility wrapper for a SARIMA baseline without exogenous features."""
    model = build_sarima({})
    try:
        model.fit(y_train)
        scores = model.predict_score(steps=steps)
    except ImportError:
        raise
    except Exception:
        # Short or degenerate series can make SARIMAX fail; use the observed
        # positive rate as a deterministic statistical fallback score.
        y_series = pd.Series(y_train).astype(float)
        rate = float(y_series.mean()) if len(y_series) else 0.0
        scores = np.full(int(steps), np.clip(rate, 0.0, 1.0), dtype=float)
    return PredictionResult(y_pred_binary=(scores >= 0.5).astype(int), y_score=scores)
