from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExperimentTracker(ABC):

    @abstractmethod
    def start_run(self, run_name: str | None = None) -> None:
        ...

    @abstractmethod
    def log_parameters(self, parameters: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None:
        ...

    @abstractmethod
    def log_artifact(self, artifact: Path) -> None:
        ...

    @abstractmethod
    def log_model(self, model: Path) -> None:
        ...

    @abstractmethod
    def end_run(self) -> None:
        ...