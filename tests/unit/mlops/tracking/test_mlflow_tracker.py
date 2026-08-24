from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.entities.experiment_metadata import ExperimentMetadata
from src.mlops.tracking import MLflowTracker

pytestmark = pytest.mark.unit


@pytest.fixture
def metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        python_version="3.12.0",
        tensorflow_version="2.18.0",
        numpy_version="2.0.0",
        operating_system="Windows",
        machine="AMD64",
        processor="test",
        git_commit="abc123",
        git_branch="develop",
        random_seed=42,
    )


@patch("src.mlops.tracking.mlflow_tracker.mlflow.start_run")
@patch("src.mlops.tracking.mlflow_tracker.mlflow.set_experiment")
@patch("src.mlops.tracking.mlflow_tracker.mlflow.set_tracking_uri")
def test_start_run(
    mock_set_tracking_uri: Mock,
    mock_set_experiment: Mock,
    mock_start_run: Mock,
) -> None:
    tracker = MLflowTracker()

    tracker.start_run("baseline")

    from src.config import settings

    mock_set_tracking_uri.assert_called_once_with(settings.mlflow_tracking_uri)
    mock_set_experiment.assert_called_once_with(settings.mlflow_experiment_name)
    mock_start_run.assert_called_once_with(run_name="baseline")


def test_log_parameters() -> None:
    tracker = MLflowTracker()

    parameters = {
        "max_iter": 1000.0,
        "random_state": 42.0,
    }

    with patch("src.mlops.tracking.mlflow_tracker.mlflow.log_params") as log_params:
        tracker.log_parameters(parameters)

    log_params.assert_called_once_with(parameters)


def test_log_metrics() -> None:
    tracker = MLflowTracker()

    metrics = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }

    with patch("src.mlops.tracking.mlflow_tracker.mlflow.log_metrics") as log_metrics:
        tracker.log_metrics(metrics)

    log_metrics.assert_called_once_with(metrics)


def test_log_artifact() -> None:
    tracker = MLflowTracker()
    artifact = Path("artifact.txt")

    with patch("src.mlops.tracking.mlflow_tracker.mlflow.log_artifact") as log_artifact:
        tracker.log_artifact(artifact)

    log_artifact.assert_called_once_with(str(artifact))


def test_log_model() -> None:
    tracker = MLflowTracker()
    model = Mock()

    with patch(
        "src.mlops.tracking.mlflow_tracker.mlflow.sklearn.log_model"
    ) as log_model:
        tracker.log_model(model)

    log_model.assert_called_once_with(
        sk_model=model,
        artifact_path="model",
    )


def test_log_metadata(
    metadata: ExperimentMetadata,
) -> None:
    tracker = MLflowTracker()

    with patch("src.mlops.tracking.mlflow_tracker.mlflow.set_tags") as set_tags:
        tracker.log_metadata(metadata)

    set_tags.assert_called_once_with(
        {
            "python_version": "3.12.0",
            "tensorflow_version": "2.18.0",
            "numpy_version": "2.0.0",
            "operating_system": "Windows",
            "machine": "AMD64",
            "processor": "test",
            "git_commit": "abc123",
            "git_branch": "develop",
            "random_seed": "42",
            "timestamp": "2026-01-01T12:00:00",
        }
    )


def test_log_metadata_handles_missing_git_information(
    metadata: ExperimentMetadata,
) -> None:
    tracker = MLflowTracker()

    metadata_without_git = ExperimentMetadata(
        timestamp=metadata.timestamp,
        python_version=metadata.python_version,
        tensorflow_version=metadata.tensorflow_version,
        numpy_version=metadata.numpy_version,
        operating_system=metadata.operating_system,
        machine=metadata.machine,
        processor=metadata.processor,
        git_commit=None,
        git_branch=None,
        random_seed=metadata.random_seed,
    )

    with patch("src.mlops.tracking.mlflow_tracker.mlflow.set_tags") as set_tags:
        tracker.log_metadata(metadata_without_git)

    tags = set_tags.call_args.args[0]

    assert tags["git_commit"] == "unknown"
    assert tags["git_branch"] == "unknown"


def test_end_run() -> None:
    tracker = MLflowTracker()

    with patch("src.mlops.tracking.mlflow_tracker.mlflow.end_run") as end_run:
        tracker.end_run()

    end_run.assert_called_once_with()
