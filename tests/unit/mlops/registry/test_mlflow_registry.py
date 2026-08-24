from unittest.mock import Mock, patch

import pytest

from src.mlops.registry import MLflowModelRegistry

pytestmark = pytest.mark.unit


def test_register_model() -> None:
    registry = MLflowModelRegistry()

    run = Mock()
    run.info.run_id = "test-run-id"

    with (
        patch(
            "src.mlops.registry.mlflow_registry.mlflow.active_run",
            return_value=run,
        ),
        patch(
            "src.mlops.registry.mlflow_registry.mlflow.register_model"
        ) as register_model,
        patch("src.mlops.registry.mlflow_registry.mlflow.set_tracking_uri"),
    ):
        registry.register_model(
            name="aegis-classifier",
        )

    register_model.assert_called_once_with(
        model_uri="runs:/test-run-id/model",
        name="aegis-classifier",
    )


def test_register_model_raises_when_no_active_run() -> None:
    registry = MLflowModelRegistry()

    with (
        patch(
            "src.mlops.registry.mlflow_registry.mlflow.active_run",
            return_value=None,
        ),
        patch("src.mlops.registry.mlflow_registry.mlflow.set_tracking_uri"),
        pytest.raises(
            RuntimeError,
            match="An active MLflow run is required to register a model.",
        ),
    ):
        registry.register_model(
            name="aegis-classifier",
        )
