import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import search_threshold, threshold_objective_value
from src.models import build_model, fit_predict_sklearn
from train import (
    ExperimentResult,
    add_oracle_features,
    add_variant_features,
    build_estimator,
    enforce_top_n,
    finalize_prepared_split,
    fill_weather_missing,
    resolve_oracle_predict_features,
    split_years,
    suggest_logreg_params,
)


class FixedElasticnetTrial:
    def suggest_categorical(self, name, choices):
        if name == "solver_penalty":
            return "saga_elasticnet"
        if name == "class_weight":
            return "balanced"
        return choices[0]

    def suggest_float(self, name, low, high, **kwargs):
        return 0.25 if name == "l1_ratio" else 0.2


class LogregThresholdTests(unittest.TestCase):
    def test_search_threshold_uses_precision_floor_then_recall(self):
        y_true = [0, 1, 1, 1]
        y_score = [0.2, 0.4, 0.8, 0.9]

        threshold, info = search_threshold(y_true, y_score)

        self.assertEqual(threshold, 0.4)
        self.assertGreaterEqual(info["precision"], 0.60)
        self.assertEqual(info["recall"], 1.0)
        self.assertEqual(info["objective"], threshold_objective_value(info["recall"], info["precision"], f1_value=info["f1"]))

    def test_search_threshold_falls_back_to_precision_recall_f1(self):
        y_true = [0, 0, 0, 1]
        y_score = [0.9, 0.8, 0.7, 0.1]

        threshold, info = search_threshold(y_true, y_score)

        self.assertEqual(threshold, 0.1)
        self.assertLess(info["precision"], 0.60)
        self.assertEqual(info["precision"], 0.25)
        self.assertEqual(info["recall"], 1.0)

    def test_expanded_logreg_params_are_used(self):
        train_estimator = build_estimator(
            "logreg",
            {
                "solver": "saga",
                "penalty": "elasticnet",
                "l1_ratio": 0.25,
                "class_weight": None,
                "C": 0.2,
            },
        )
        train_clf = train_estimator.named_steps["model"]
        src_clf = build_model("logreg", {"solver_penalty": "liblinear_l1"}).named_steps["clf"]

        self.assertEqual(train_clf.solver, "saga")
        self.assertEqual(train_clf.penalty, "elasticnet")
        self.assertEqual(train_clf.l1_ratio, 0.25)
        self.assertEqual(src_clf.solver, "liblinear")
        self.assertEqual(src_clf.penalty, "l1")

    def test_logreg_default_solver_matches_notebook(self):
        estimator = build_estimator("logreg")

        self.assertEqual(estimator.named_steps["model"].solver, "liblinear")

    def test_logreg_tuning_space_derives_valid_solver_penalty(self):
        params = suggest_logreg_params(FixedElasticnetTrial())

        self.assertEqual(params["solver"], "saga")
        self.assertEqual(params["penalty"], "elasticnet")
        self.assertEqual(params["l1_ratio"], 0.25)
        self.assertEqual(params["class_weight"], "balanced")

    def test_logreg_fit_predict_applies_validation_threshold_to_test_scores(self):
        x_train = pd.DataFrame({"signal": [0.0, 0.5, 1.0, 3.0, 3.5, 4.0]})
        y_train = pd.Series([0, 0, 0, 1, 1, 1])
        x_val = pd.DataFrame({"signal": [0.25, 2.5, 3.75]})
        y_val = pd.Series([0, 1, 1])
        x_test = pd.DataFrame({"signal": [0.1, 2.0, 4.2]})

        result = fit_predict_sklearn("logreg", x_train, y_train, x_test, X_val=x_val, y_val=y_val)

        self.assertIsNotNone(result.y_score)
        self.assertIsNotNone(result.validation_metrics)
        self.assertIsNotNone(result.threshold_objective)
        self.assertGreaterEqual(result.threshold, 0.0)
        self.assertLessEqual(result.threshold, 1.0)
        np.testing.assert_array_equal(result.y_pred_binary, (result.y_score >= result.threshold).astype(int))

    def test_split_years_matches_notebook_six_validation_six_test_years(self):
        years = [
            1972, 1973, 1977, 1982, 1983, 1984, 1987, 1991, 1992, 1993,
            1994, 1995, 1996, 1997, 1998, 2002, 2004, 2005, 2006, 2007,
            2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
        ]
        df = pd.DataFrame({"year": years, "day": 1, "target_favorable": 0})

        train_years, val_years, test_years = split_years(df)

        self.assertEqual(train_years, years[:18])
        self.assertEqual(val_years, years[18:24])
        self.assertEqual(test_years, years[24:])

    def test_fill_weather_missing_reuses_train_medians_without_backfill(self):
        train_df = pd.DataFrame({
            "year": [2000, 2000, 2001],
            "day": [1, 2, 1],
            "t_min": [np.nan, 5.0, 7.0],
            "is_rain": [np.nan, 1.0, np.nan],
        })
        val_df = pd.DataFrame({
            "year": [2002],
            "day": [1],
            "t_min": [np.nan],
            "is_rain": [np.nan],
        })

        train_filled, fill_values = fill_weather_missing(train_df)
        val_filled, _ = fill_weather_missing(val_df, fill_values)

        self.assertEqual(train_filled.loc[0, "t_min"], 6.0)
        self.assertEqual(fill_values["t_min"], 6.0)
        self.assertEqual(val_filled.loc[0, "t_min"], 6.0)
        self.assertEqual(train_filled.loc[0, "is_rain"], 0.0)
        self.assertEqual(val_filled.loc[0, "is_rain"], 0.0)

    def test_temporal_feature_set_matches_best_logreg_notebook(self):
        import train

        self.assertEqual(train.TEMPORAL_LAG_STEPS, [2, 7])
        self.assertEqual(train.TEMPORAL_ROLLING_WINDOWS, [3])

    def test_oracle_features_shift_within_year_and_leave_tail_nan(self):
        df = pd.DataFrame({
            "year": [2000, 2000, 2000, 2001, 2001],
            "day": [1, 2, 3, 1, 2],
            "t_min": [10.0, 11.0, 12.0, 20.0, 21.0],
            "target_favorable": [0, 1, 0, 1, 0],
        })

        shifted, oracle_cols = add_oracle_features(df, 1, ["t_min"])

        self.assertEqual(oracle_cols, ["oracle_t_min_h1"])
        self.assertEqual(shifted.loc[0, "oracle_t_min_h1"], 11.0)
        self.assertEqual(shifted.loc[1, "oracle_t_min_h1"], 12.0)
        self.assertTrue(np.isnan(shifted.loc[2, "oracle_t_min_h1"]))
        self.assertEqual(shifted.loc[3, "oracle_t_min_h1"], 21.0)
        self.assertTrue(np.isnan(shifted.loc[4, "oracle_t_min_h1"]))

    def test_oracle_tail_rows_are_removed_by_existing_finalize_step(self):
        df = pd.DataFrame({
            "year": [2000, 2000, 2000],
            "day": [1, 2, 3],
            "target_h1": [1, 0, np.nan],
            "t_min": [10.0, 11.0, 12.0],
        })
        shifted, oracle_cols = add_oracle_features(df, 1, ["t_min"])

        finalized = finalize_prepared_split(shifted, ["t_min", *oracle_cols], "target_h1")

        self.assertEqual(finalized["day"].tolist(), [1, 2])
        self.assertEqual(finalized["oracle_t_min_h1"].tolist(), [11.0, 12.0])

    def test_oracle_predict_features_reject_targets_and_non_weather_columns(self):
        columns = ["year", "day", "t_min", "target_favorable", "target_h2"]

        with self.assertRaises(ValueError):
            resolve_oracle_predict_features(["target_favorable"], columns)
        with self.assertRaises(ValueError):
            resolve_oracle_predict_features(["target_h2"], columns)
        with self.assertRaises(ValueError):
            resolve_oracle_predict_features(["year"], columns)
        with self.assertRaises(ValueError):
            resolve_oracle_predict_features(["day"], columns)

    def test_oracle_predict_features_use_default_weather_list(self):
        columns = [
            "year",
            "day",
            "target_favorable",
            "target_h2",
            "t_min",
            "t_max",
            "t_avg",
            "precipitation",
            "is_rain",
            "cloudiness",
            "y1",
            "y2",
            "y2_y1",
            "y3",
            "y4",
            "precipitation_t_gt_10",
            "t_gt_10",
        ]

        features = resolve_oracle_predict_features(None, columns)

        self.assertIn("t_min", features)
        self.assertIn("t_gt_10", features)
        self.assertNotIn("target_favorable", features)
        self.assertNotIn("target_h2", features)
        self.assertNotIn("year", features)
        self.assertNotIn("day", features)

    def test_normal_variant_features_do_not_include_oracle_columns(self):
        df = pd.DataFrame({
            "year": [2000, 2000, 2000],
            "day": [1, 2, 3],
            "t_min": [10.0, 11.0, 12.0],
            "t_max": [20.0, 21.0, 22.0],
            "is_rain": [0, 1, 0],
            "target_favorable": [0, 1, 0],
            "y2": [1.0, 1.0, 1.0],
            "y1": [1.0, 1.0, 1.0],
            "y2_y1": [0.0, 0.0, 0.0],
            "precipitation": [0.0, 1.0, 0.0],
            "t_avg": [15.0, 16.0, 17.0],
            "cloudiness": [3.0, 4.0, 5.0],
            "y3": [0.0, 0.0, 0.0],
            "y4": [0.0, 0.0, 0.0],
            "precipitation_t_gt_10": [0.0, 1.0, 0.0],
            "t_gt_10": [15.0, 16.0, 17.0],
        })

        frame, feature_cols = add_variant_features(df, 7, "baseline")

        self.assertFalse(any(col.startswith("oracle_") for col in frame.columns))
        self.assertFalse(any(col.startswith("oracle_") for col in feature_cols))

    def test_top_n_uses_notebook_rank_before_f1(self):
        low_precision_high_f1 = ExperimentResult(
            "logreg", "baseline", 2, 7, 0.95, 0.50, 0.95, 0.99, 0.99, Path("missing-a"), Path("missing-a.zip")
        )
        eligible_lower_recall = ExperimentResult(
            "logreg", "interaction_fe", 2, 7, 0.40, 0.60, 0.70, 0.80, 0.80, Path("missing-b"), Path("missing-b.zip")
        )
        eligible_higher_recall = ExperimentResult(
            "logreg", "temporal_fe", 2, 7, 0.45, 0.61, 0.80, 0.70, 0.70, Path("missing-c"), Path("missing-c.zip")
        )

        kept = enforce_top_n([low_precision_high_f1, eligible_lower_recall, eligible_higher_recall], top_n=2)

        self.assertEqual([row.variant for row in kept], ["temporal_fe", "interaction_fe"])


if __name__ == "__main__":
    unittest.main()
