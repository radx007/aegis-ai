from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.config import settings
from src.dataset import Dataset
from src.exceptions.dataset import DatasetError

pytestmark = pytest.mark.unit


def test_load_returns_arrays(y_true: np.ndarray, dummy_features: np.ndarray) -> None:

    with patch(
        "src.dataset.dataset.np.load",
        side_effect=[dummy_features, y_true],
    ):
        dataset = Dataset(settings.processed_data_path)

        loaded_X, loaded_y = dataset.load()

    assert np.array_equal(
        loaded_X,
        dummy_features,
    )

    assert np.array_equal(
        loaded_y,
        y_true,
    )


def test_load_raises_dataset_error() -> None:

    with patch(
        "src.dataset.dataset.np.load",
        side_effect=Exception,
    ):
        dataset = Dataset(settings.processed_data_path)

        with pytest.raises(DatasetError):
            dataset.load()


def test_split_returns_four_arrays(monkeypatch: pytest.MonkeyPatch) -> None:

    dataset = Dataset(settings.processed_data_path)

    monkeypatch.setattr(
        dataset,
        "load",
        Mock(
            return_value=(
                np.random.rand(20, 1024),
                np.array([0] * 20),
            )
        ),
    )

    X_train, X_test, y_train, y_test = dataset.split()

    assert len(X_train) == 16
    assert len(X_test) == 4

    assert len(y_train) == 16
    assert len(y_test) == 4


def test_split_raises_dataset_error(monkeypatch: pytest.MonkeyPatch) -> None:

    dataset = Dataset(settings.processed_data_path)

    monkeypatch.setattr(
        dataset,
        "load",
        Mock(side_effect=DatasetError("Unable to load processed dataset.")),
    )

    with pytest.raises(DatasetError):
        dataset.split()
