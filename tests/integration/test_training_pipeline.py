from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.dataset import Dataset
from src.evaluation import Evaluator
from src.models import ModelRepository
from src.training import Trainer

pytestmark = pytest.mark.integration


def test_training_pipeline(
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

    np.save(
        tmp_path / "X.npy",
        X,
    )

    np.save(
        tmp_path / "y.npy",
        y,
    )

    dataset = Dataset(
        tmp_path,
    )

    repository = ModelRepository(
        tmp_path / "baseline.pkl",
    )

    evaluator = Evaluator()

    trainer = Trainer(
        dataset=dataset,
        evaluator=evaluator,
        repository=repository,
    )

    # Act
    result = trainer.train()

    loaded_model = repository.load()

    # Assert
    assert result.model_path.exists()

    assert result.metrics.accuracy >= 0.0
    assert result.metrics.precision >= 0.0
    assert result.metrics.recall >= 0.0
    assert result.metrics.f1 >= 0.0

    assert loaded_model is not None
