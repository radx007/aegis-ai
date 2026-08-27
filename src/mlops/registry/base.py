from abc import ABC, abstractmethod


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
