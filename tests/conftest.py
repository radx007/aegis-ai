from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.entities.metrics import EvaluationMetrics
from src.evaluation import Evaluator


@pytest.fixture
def evaluator() -> Evaluator:
    return Evaluator()


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([0, 1, 2, 3])


@pytest.fixture
def y_pred() -> np.ndarray:
    return np.array([0, 1, 2, 3])


@pytest.fixture
def mock_dataset() -> Mock:
    mock = Mock()
    mock.split.return_value = (
        np.random.rand(10, 1024),
        np.random.rand(2, 1024),
        np.array([0, 1] * 5),
        np.array([0, 1]),
    )
    return mock


@pytest.fixture
def mock_repository() -> Mock:
    mock = Mock()
    mock.save.return_value = Path("baseline.pkl")
    return mock


@pytest.fixture
def mock_evaluator() -> Mock:
    mock = Mock()
    mock.evaluate.return_value = EvaluationMetrics(
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        confusion_matrix=np.eye(2),
    )
    return mock
