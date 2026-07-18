from pathlib import Path
from unittest.mock import Mock

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.embeddings import EmbeddingExtractor
from src.inference import Predictor
from src.models import ModelRepository


def test_prediction_pipeline(
    tmp_path: Path,
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

    model.fit(
        X,
        y,
    )

    repository = ModelRepository(
        tmp_path / "baseline.pkl",
    )

    repository.save(
        model,
    )

    loaded_model = repository.load()

    extractor = Mock(
        spec=EmbeddingExtractor,
    )

    extractor.extract.return_value = X[0]

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

    extractor.extract.assert_called_once_with(
        Path("audio.wav"),
    )
