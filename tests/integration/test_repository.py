from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.models import ModelRepository

pytestmark = pytest.mark.integration


def test_repository_round_trip(
    tmp_path: Path,
) -> None:

    # Arrange
    X, y = make_classification(
        n_samples=20,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    model = LogisticRegression()

    model.fit(
        X,
        y,
    )

    repository = ModelRepository(
        tmp_path / "baseline.pkl",
    )

    # Act
    saved_path = repository.save(
        model,
    )

    loaded_model = repository.load()

    # Assert
    assert saved_path.exists()

    assert isinstance(
        loaded_model,
        LogisticRegression,
    )

    original_predictions = model.predict(
        X,
    )

    loaded_predictions = loaded_model.predict(
        X,
    )

    assert np.array_equal(
        original_predictions,
        loaded_predictions,
    )
