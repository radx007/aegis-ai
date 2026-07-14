from pathlib import Path
from unittest.mock import Mock

from src.entities import TrainingResult
from src.training import Trainer


def test_train_returns_training_result(
    mock_dataset: Mock,
    mock_repository: Mock,
    mock_evaluator: Mock,
) -> None:

    trainer = Trainer(
        dataset=mock_dataset,
        repository=mock_repository,
        evaluator=mock_evaluator,
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
