from unittest.mock import Mock, patch

import pytest

from src.config import settings
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
        patch("src.mlops.registry.mlflow_registry.mlflow.set_tracking_uri") as set_uri,
    ):
        registry.register_model(
            name="aegis-classifier",
        )

    set_uri.assert_called_once_with(settings.mlflow_tracking_uri)
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


def test_promote_model() -> None:
    registry = MLflowModelRegistry()

    mock_client_instance = Mock()

    with (
        patch("src.mlops.registry.mlflow_registry.mlflow.set_tracking_uri") as set_uri,
        patch(
            "src.mlops.registry.mlflow_registry.mlflow.MlflowClient",
            return_value=mock_client_instance,
        ) as mock_client_cls,
    ):
        registry.promote(
            name="aegis-classifier",
            version="8",
            alias="champion",
        )

    set_uri.assert_called_once_with(settings.mlflow_tracking_uri)
    mock_client_cls.assert_called_once_with(tracking_uri=settings.mlflow_tracking_uri)
    mock_client_instance.set_registered_model_alias.assert_called_once_with(
        name="aegis-classifier",
        alias="champion",
        version="8",
    )


def test_get_model_by_alias() -> None:
    registry = MLflowModelRegistry()

    client = Mock()

    model_version = Mock()
    model_version.name = "aegis-classifier"
    model_version.version = "8"

    client.get_model_version_by_alias.return_value = model_version

    with (
        patch("src.mlops.registry.mlflow_registry.mlflow.set_tracking_uri"),
        patch(
            "src.mlops.registry.mlflow_registry.mlflow.MlflowClient",
            return_value=client,
        ),
    ):
        result = registry.get_model_by_alias(
            name="aegis-classifier",
            alias="champion",
        )

    client.get_model_version_by_alias.assert_called_once_with(
        name="aegis-classifier",
        alias="champion",
    )

    assert result.name == "aegis-classifier"
    assert result.version == "8"
