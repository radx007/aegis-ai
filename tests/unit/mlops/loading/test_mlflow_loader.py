from unittest.mock import Mock, patch

import pytest

from src.config import settings
from src.entities.registered_model import RegisteredModelVersion
from src.mlops.loading import MLflowModelLoader

pytestmark = pytest.mark.unit


def test_load_model() -> None:
    loader = MLflowModelLoader()

    model = RegisteredModelVersion(
        name=settings.mlflow_model_name,
        version="8",
    )

    loaded_model = Mock()

    with (
        patch("src.mlops.loading.mlflow_loader.mlflow.set_tracking_uri"),
        patch(
            "src.mlops.loading.mlflow_loader.mlflow.sklearn.load_model",
            return_value=loaded_model,
        ) as load_model,
    ):
        result = loader.load(model)

    load_model.assert_called_once_with(
        "models:/aegis-classifier/8",
    )

    assert result is loaded_model
