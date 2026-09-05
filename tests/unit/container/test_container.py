from unittest.mock import patch

import pytest

from src.container import Container
from src.exceptions import ModelLoadingError
from src.mlops.registry import (
    MLflowModelRegistry,
    NullModelRegistry,
)
from src.mlops.tracking import (
    MLflowTracker,
    NullTracker,
)

pytestmark = pytest.mark.unit


def test_uses_mlflow_adapters_when_mlflow_enabled() -> None:
    with patch("src.container.settings.mlflow_enabled", True):
        container = Container()

        assert isinstance(
            container.tracker,
            MLflowTracker,
        )

        assert isinstance(
            container.registry,
            MLflowModelRegistry,
        )


def test_uses_null_adapters_when_mlflow_disabled() -> None:
    with patch("src.container.settings.mlflow_enabled", False):
        container = Container()

        assert isinstance(
            container.tracker,
            NullTracker,
        )

        assert isinstance(
            container.registry,
            NullModelRegistry,
        )


def test_model_requires_mlflow() -> None:
    with (
        patch("src.container.settings.mlflow_enabled", False),
        pytest.raises(
            ModelLoadingError,
            match="MLflow must be enabled to load the production model.",
        ),
    ):
        _ = Container().model
