from pathlib import Path
from typing import Any

import mlflow
from sklearn.base import ClassifierMixin

from src.entities import ExperimentMetadata

from .base import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    def __init__(self, experiment_name: str) -> None:
        self._experiment_name = experiment_name

    def start_run(self, run_name: str | None = None) -> None:
        mlflow.set_experiment(self._experiment_name)
        mlflow.start_run(run_name=run_name)

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        mlflow.log_params(parameters)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact(self, artifact: Path) -> None:
        mlflow.log_artifact(str(artifact))

    def log_model(self, model: ClassifierMixin) -> None:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

    def log_metadata(
        self,
        metadata: ExperimentMetadata,
    ) -> None:
        tags = {
            "python_version": metadata.python_version,
            "tensorflow_version": metadata.tensorflow_version,
            "numpy_version": metadata.numpy_version,
            "operating_system": metadata.operating_system,
            "machine": metadata.machine,
            "processor": metadata.processor,
            "git_commit": metadata.git_commit or "unknown",
            "git_branch": metadata.git_branch or "unknown",
            "random_seed": str(metadata.random_seed),
            "timestamp": metadata.timestamp.isoformat(),
        }

        mlflow.set_tags(tags)

    def end_run(self) -> None:
        mlflow.end_run()
