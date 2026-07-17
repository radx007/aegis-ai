from unittest.mock import Mock, patch

import pytest

from src.config import settings
from src.exceptions.model import ModelError
from src.models import ModelRepository


def test_save_returns_model_path(mock_model: Mock) -> None:

    repository = ModelRepository(settings.baseline_model_path)

    with patch("src.models.repository.joblib.dump"):
        path = repository.save(mock_model)

    assert path == settings.baseline_model_path


def test_load_returns_model(mock_model: Mock) -> None:

    repository = ModelRepository(settings.baseline_model_path)

    with patch(
        "src.models.repository.joblib.load",
        return_value=mock_model,
    ):
        loaded = repository.load()

    assert loaded is mock_model


def test_save_raises_model_error(mock_model: Mock) -> None:

    repository = ModelRepository(settings.baseline_model_path)

    with patch(
        "src.models.repository.joblib.dump",
        side_effect=Exception,
    ):
        with pytest.raises(ModelError):
            repository.save(mock_model)


def test_load_raises_model_error() -> None:

    repository = ModelRepository(settings.baseline_model_path)

    with patch(
        "src.models.repository.joblib.load",
        side_effect=Exception,
    ):
        with pytest.raises(ModelError):
            repository.load()
