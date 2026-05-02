import unittest

import numpy as np
import pandas as pd

from src.evaluation import search_threshold, threshold_objective_value
from src.models import fit_predict_sklearn


class LogregThresholdTests(unittest.TestCase):
    def test_search_threshold_uses_precision_floor_then_recall(self):
        y_true = [0, 1, 1, 1]
        y_score = [0.2, 0.4, 0.8, 0.9]

        threshold, info = search_threshold(y_true, y_score)

        self.assertEqual(threshold, 0.4)
        self.assertGreaterEqual(info["precision"], 0.60)
        self.assertEqual(info["recall"], 1.0)
        self.assertEqual(info["objective"], threshold_objective_value(info["recall"], info["precision"]))

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


if __name__ == "__main__":
    unittest.main()
