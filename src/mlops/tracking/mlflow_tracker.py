from pathlib import Path
from typing import Any

from src.entities.experiment_metadata import ExperimentMetadata

from .base import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    def start_run(self, run_name: str | None = None) -> None:
        raise NotImplementedError

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: dict[str, float]) -> None:
        raise NotImplementedError

    def log_artifact(self, artifact: Path) -> None:
        raise NotImplementedError

    def log_model(self, model: Path) -> None:
        raise NotImplementedError

    def log_metadata(
        self,
        metadata: ExperimentMetadata,
    ) -> None:
        raise NotImplementedError("MLflow integration has not been implemented yet.")

    def end_run(self) -> None:
        raise NotImplementedError
