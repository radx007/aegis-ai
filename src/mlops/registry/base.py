from abc import ABC, abstractmethod

from src.entities.registered_model import RegisteredModelVersion


class ModelRegistry(ABC):
    @abstractmethod
    def register_model(
        self,
        name: str,
    ) -> None: ...

    @abstractmethod
    def promote(
        self,
        name: str,
        version: str,
        alias: str,
    ) -> None: ...

    @abstractmethod
    def get_model_by_alias(
        self,
        name: str,
        alias: str,
    ) -> RegisteredModelVersion: ...
