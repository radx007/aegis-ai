from pathlib import Path
from typing import Any

from .base import ExperimentTracker


class NullTracker(ExperimentTracker):

    def start_run(self, run_name: str | None = None) -> None:
        return None

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float]) -> None:
        return None

    def log_artifact(self, artifact: Path) -> None:
        return None

    def log_model(self, model: Path) -> None:
        return None

    def end_run(self) -> None:
        return None