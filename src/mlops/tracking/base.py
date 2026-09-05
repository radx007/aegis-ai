from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sklearn.base import ClassifierMixin

from src.entities import ExperimentMetadata


class ExperimentTracker(ABC):
    @abstractmethod
    def start_run(self, run_name: str | None = None) -> None: ...

    @abstractmethod
    def log_parameters(self, parameters: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    @abstractmethod
    def log_artifact(self, artifact: Path) -> None: ...

    @abstractmethod
    def log_model(self, model: ClassifierMixin) -> None: ...

    @abstractmethod
    def log_metadata(self, metadata: ExperimentMetadata) -> None: ...

    @abstractmethod
    def end_run(self) -> None: ...
