from unittest.mock import Mock, patch

import pytest

from src.config import settings
from src.mlops.registry import MLflowModelRegistry

pytestmark = pytest.mark.unit


def test_register_model() -> None:
    registry = MLflowModelRegistry()

    run = Mock()
    run.info.run_id = "test-run-id"

    registered_version = Mock()
    registered_version.name = settings.mlflow_model_name
    registered_version.version = "9"

    with (
        patch(
            "src.mlops.registry.mlflow_registry.mlflow.active_run",
            return_value=run,
        ),
        patch(
            "src.mlops.registry.mlflow_registry.mlflow.register_model",
            return_value=registered_version,
        ) as register_model,
        patch("src.mlops.registry.mlflow_registry.mlflow.set_tracking_uri"),
    ):
        result = registry.register_model(
            name=settings.mlflow_model_name,
        )

    register_model.assert_called_once_with(
        model_uri="runs:/test-run-id/model",
        name=settings.mlflow_model_name,
    )

    assert result.name == settings.mlflow_model_name
    assert result.version == "9"


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
            name=settings.mlflow_model_name,
        )


def test_promote_model() -> None:
    registry = MLflowModelRegistry()

    mock_client_instance = Mock()

    with (
        patch("src.mlops.registry.mlflow_registry.mlflow.set_tracking_uri"),
        patch(
            "src.mlops.registry.mlflow_registry.mlflow.MlflowClient",
            return_value=mock_client_instance,
        ) as mock_client_cls,
    ):
        registry.promote(
            name=settings.mlflow_model_name,
            version="8",
            alias=settings.mlflow_model_alias,
        )

    mock_client_cls.assert_called_once_with()
    mock_client_instance.set_registered_model_alias.assert_called_once_with(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
        version="8",
    )


def test_get_model_by_alias() -> None:
    registry = MLflowModelRegistry()

    client = Mock()

    model_version = Mock()
    model_version.name = settings.mlflow_model_name
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
            name=settings.mlflow_model_name,
            alias=settings.mlflow_model_alias,
        )

    client.get_model_version_by_alias.assert_called_once_with(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
    )

    assert result.name == settings.mlflow_model_name
    assert result.version == "8"
