from pathlib import Path
from unittest.mock import Mock

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.embeddings import EmbeddingModel
from src.inference import Predictor
from src.mlops.loading import ModelLoader

pytestmark = pytest.mark.integration


def test_prediction_pipeline(
    tmp_path: Path,
    registry: Mock,
) -> None:
    # Arrange
    X, y = make_classification(
        n_samples=50,
        n_features=1024,
        n_informative=20,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    model = LogisticRegression(
        max_iter=1000,
    )
    model.fit(X, y)

    loader = Mock(spec=ModelLoader)

    registered_model = Mock()
    registered_model.name = settings.mlflow_model_name
    registered_model.version = "8"

    registry.get_model_by_alias.return_value = registered_model
    loader.load.return_value = model

    extractor = Mock(spec=EmbeddingModel)
    extractor.extract.return_value = X[0]

    # Resolve model through Registry → Loader
    resolved_model = registry.get_model_by_alias(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
    )

    loaded_model = loader.load(resolved_model)

    predictor = Predictor(
        model=loaded_model,
        extractor=extractor,
    )

    # Act
    result = predictor.predict(
        Path("audio.wav"),
    )

    # Assert
    assert result.label in {"0", "1"}
    assert 0.0 <= result.confidence <= 1.0

    registry.get_model_by_alias.assert_called_once_with(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
    )

    loader.load.assert_called_once_with(
        registered_model,
    )

    extractor.extract.assert_called_once_with(
        Path("audio.wav"),
    )
