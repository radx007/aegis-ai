from src.entities.registered_model import RegisteredModelVersion

from .base import ModelRegistry


class NullModelRegistry(ModelRegistry):
    def register_model(
        self,
        name: str,
    ) -> RegisteredModelVersion:
        return RegisteredModelVersion(
            name=name,
            version="",
        )

    def promote(
        self,
        name: str,
        version: str,
        alias: str,
    ) -> None:
        return None

    def get_model_by_alias(
        self,
        name: str,
        alias: str,
    ) -> RegisteredModelVersion:
        return RegisteredModelVersion(
            name=name,
            version="",
        )
