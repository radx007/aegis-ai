from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin

from src.entities import RegisteredModelVersion


class ModelLoader(ABC):
    @abstractmethod
    def load(
        self,
        model: RegisteredModelVersion,
    ) -> ClassifierMixin: ...
