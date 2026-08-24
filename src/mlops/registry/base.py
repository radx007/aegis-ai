from abc import ABC, abstractmethod


class ModelRegistry(ABC):
    @abstractmethod
    def register_model(
        self,
        name: str,
    ) -> None: ...
