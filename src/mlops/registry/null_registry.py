from pathlib import Path

from .base import ModelRegistry


class NullModelRegistry(ModelRegistry):
    def register_model(
        self,
        model: Path,
        name: str,
    ) -> None:
        return None