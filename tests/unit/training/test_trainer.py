from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.entities import TrainingResult
from src.entities.experiment_metadata import ExperimentMetadata
from src.exceptions.training import TrainingError
from src.training import Trainer

pytestmark = pytest.mark.unit


def test_train_returns_training_result(
    mock_dataset: Mock,
    mock_repository: Mock,
    mock_evaluator: Mock,
    tracker: Mock,
    metadata_collector: Mock,
) -> None:

    trainer = Trainer(
        dataset=mock_dataset,
        repository=mock_repository,
        evaluator=mock_evaluator,
        tracker=tracker,
        metadata_collector=metadata_collector,
    )

    result = trainer.train()

    assert isinstance(
        result,
        TrainingResult,
    )

    assert result.metrics.accuracy == 1.0

    assert result.model_path == Path("baseline.pkl")

    mock_dataset.split.assert_called_once()

    mock_repository.save.assert_called_once()

    mock_evaluator.evaluate.assert_called_once()


def test_train_raises_training_error_when_model_fit_fails(
    mock_dataset: Mock,
    mock_repository: Mock,
    mock_evaluator: Mock,
    tracker: Mock,
    metadata_collector: Mock,
) -> None:

    trainer = Trainer(
        dataset=mock_dataset,
        repository=mock_repository,
        evaluator=mock_evaluator,
        tracker=tracker,
        metadata_collector=metadata_collector,
    )

    from unittest.mock import patch

    with patch("src.training.train.LogisticRegression.fit", side_effect=Exception):
        with pytest.raises(TrainingError):
            trainer.train()

    mock_repository.save.assert_not_called()

    mock_evaluator.evaluate.assert_not_called()


def test_train_logs_experiment_tracking(
    mock_dataset: Mock,
    mock_repository: Mock,
    mock_evaluator: Mock,
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

    metadata_collector.collect.return_value = metadata

    trainer = Trainer(
        dataset=mock_dataset,
        repository=mock_repository,
        evaluator=mock_evaluator,
        tracker=tracker,
        metadata_collector=metadata_collector,
    )
    trainer.train()

    tracker.start_run.assert_called_once_with("baseline")
    metadata_collector.collect.assert_called_once_with()
    tracker.log_metadata.assert_called_once_with(metadata)
    tracker.log_parameters.assert_called_once()
    tracker.log_metrics.assert_called_once()
    tracker.log_model.assert_called_once()
    tracker.end_run.assert_called_once()
