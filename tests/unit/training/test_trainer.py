from datetime import datetime
from unittest.mock import Mock

import pytest

from src.config import settings
from src.entities import TrainingResult
from src.entities.experiment_metadata import ExperimentMetadata
from src.entities.registered_model import RegisteredModelVersion
from src.exceptions.training import TrainingError
from src.training import Trainer

pytestmark = pytest.mark.unit


def test_train_returns_training_result(
    mock_dataset: Mock,
    mock_evaluator: Mock,
    tracker: Mock,
    metadata_collector: Mock,
    registry: Mock,
) -> None:

    trainer = Trainer(
        dataset=mock_dataset,
        evaluator=mock_evaluator,
        tracker=tracker,
        metadata_collector=metadata_collector,
        registry=registry,
    )

    result = trainer.train()

    assert isinstance(
        result,
        TrainingResult,
    )

    assert result.metrics.accuracy == 1.0

    mock_dataset.split.assert_called_once()

    mock_evaluator.evaluate.assert_called_once()


def test_train_raises_training_error_when_model_fit_fails(
    mock_dataset: Mock,
    mock_evaluator: Mock,
    tracker: Mock,
    metadata_collector: Mock,
    registry: Mock,
) -> None:

    trainer = Trainer(
        dataset=mock_dataset,
        evaluator=mock_evaluator,
        tracker=tracker,
        metadata_collector=metadata_collector,
        registry=registry,
    )

    from unittest.mock import patch

    with patch("src.training.train.LogisticRegression.fit", side_effect=Exception):
        with pytest.raises(TrainingError):
            trainer.train()

    mock_evaluator.evaluate.assert_not_called()


def test_train_logs_experiment_tracking(
    mock_dataset: Mock,
    mock_evaluator: Mock,
    registry: Mock,
) -> None:

    tracker = Mock()
    metadata_collector = Mock()
    metadata = ExperimentMetadata(
        timestamp=datetime.now(),
        python_version="3.12.0",
        tensorflow_version="2.18.0",
        numpy_version="2.0.0",
        operating_system="Windows",
        machine="AMD64",
        processor="test",
        git_commit=None,
        git_branch=None,
        random_seed=42,
    )

    registry.register_model.return_value = RegisteredModelVersion(
        name=settings.mlflow_model_name,
        version="9",
    )

    metadata_collector.collect.return_value = metadata

    trainer = Trainer(
        dataset=mock_dataset,
        evaluator=mock_evaluator,
        tracker=tracker,
        metadata_collector=metadata_collector,
        registry=registry,
    )
    trainer.train()

    tracker.start_run.assert_called_once_with(settings.Mlflow_run_name)
    metadata_collector.collect.assert_called_once_with()
    tracker.log_metadata.assert_called_once_with(metadata)
    tracker.log_parameters.assert_called_once()
    tracker.log_metrics.assert_called_once()
    tracker.log_model.assert_called_once()
    tracker.end_run.assert_called_once()

    registry.register_model.assert_called_once_with(
        name=settings.mlflow_model_name,
    )

    registered_model = registry.register_model.return_value

    registry.promote.assert_called_once_with(
        name=registered_model.name,
        version=registered_model.version,
        alias=settings.mlflow_model_alias,
    )
