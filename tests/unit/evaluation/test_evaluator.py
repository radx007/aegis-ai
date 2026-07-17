import numpy as np

from src.entities import EvaluationMetrics
from src.evaluation import Evaluator


def test_evaluate_returns_metrics(
    evaluator: Evaluator,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    assert isinstance(
        result,
        EvaluationMetrics,
    )


def test_perfect_prediction_metrics(
    evaluator: Evaluator,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_confusion_matrix_shape(
    evaluator: Evaluator,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    assert result.confusion_matrix.shape == (
        4,
        4,
    )
