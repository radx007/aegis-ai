import mlflow.sklearn
from sklearn.base import ClassifierMixin

from src.entities import RegisteredModelVersion

from .base import ModelLoader


class MLflowModelLoader(ModelLoader):
    def load(
        self,
        model: RegisteredModelVersion,
    ) -> ClassifierMixin:
        model_uri = f"models:/{model.name}/{model.version}"

        return mlflow.sklearn.load_model(model_uri)
