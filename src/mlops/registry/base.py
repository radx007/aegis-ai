from abc import ABC, abstractmethod
from pathlib import Path


class ModelRegistry(ABC):
    @abstractmethod
    def register_model(
        self,
        model: Path,
        name: str,
    ) -> None: ...