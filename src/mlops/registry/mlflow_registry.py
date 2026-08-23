from pathlib import Path

from .base import ModelRegistry


class MLflowModelRegistry(ModelRegistry):
    def register_model(
        self,
        model: Path,
        name: str,
    ) -> None:
        raise NotImplementedError