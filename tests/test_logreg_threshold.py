import numpy as np
import pandas as pd

from src.evaluation import search_threshold, threshold_objective_value
from src.models import fit_predict_sklearn


def test_search_threshold_uses_validation_f1():
    y_true = [0, 1, 1, 1]
    y_score = [0.2, 0.4, 0.8, 0.9]

    threshold, objective = search_threshold(y_true, y_score)

    assert threshold == 0.4
    assert objective == 1.0
    assert threshold_objective_value(y_true, y_score, threshold) == 1.0


def test_logreg_fit_predict_applies_validation_threshold_to_test_scores():
    x_train = pd.DataFrame({"signal": [0.0, 0.5, 1.0, 3.0, 3.5, 4.0]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    x_val = pd.DataFrame({"signal": [0.25, 2.5, 3.75]})
    y_val = pd.Series([0, 1, 1])
    x_test = pd.DataFrame({"signal": [0.1, 2.0, 4.2]})

    result = fit_predict_sklearn("logreg", x_train, y_train, x_test, X_val=x_val, y_val=y_val)

    assert result.y_score is not None
    assert result.validation_metrics is not None
    assert result.threshold_objective is not None
    assert 0.0 <= result.threshold <= 1.0
    np.testing.assert_array_equal(result.y_pred_binary, (result.y_score >= result.threshold).astype(int))
