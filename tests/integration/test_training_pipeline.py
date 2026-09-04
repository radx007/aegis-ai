from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.config import settings
from src.dataset import Dataset
from src.entities.registered_model import RegisteredModelVersion
from src.evaluation import Evaluator
from src.training import Trainer
from src.training.config import TrainingConfig

pytestmark = pytest.mark.integration


def test_training_pipeline(
    tmp_path: Path,
    metadata_collector: Mock,
    tracker: Mock,
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

    evaluator = Evaluator()

    registry.register_model.return_value = RegisteredModelVersion(
        name=settings.mlflow_model_name,
        version="28",
    )

    config = TrainingConfig(
        run_name=settings.mlflow_run_name,
        max_iter=100,
        random_state=42,
        model_name=settings.mlflow_model_name,
        model_alias=settings.mlflow_model_alias,
        test_size=0.2,
    )

    trainer = Trainer(
        dataset=dataset,
        evaluator=evaluator,
        tracker=tracker,
        metadata_collector=metadata_collector,
        registry=registry,
        config=config,
    )

    # Act
    result = trainer.train()

    registry.register_model.assert_called_once_with(
        name=settings.mlflow_model_name,
    )

    registry.promote.assert_called_once_with(
        name=settings.mlflow_model_name,
        version="28",
        alias=settings.mlflow_model_alias,
    )

    # Assert
    assert result.metrics.accuracy >= 0.0
    assert result.metrics.precision >= 0.0
    assert result.metrics.recall >= 0.0
    assert result.metrics.f1 >= 0.0
