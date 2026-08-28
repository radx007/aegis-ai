import mlflow.sklearn
from sklearn.base import ClassifierMixin

from src.config import settings
from src.entities.registered_model import RegisteredModelVersion

from .base import ModelLoader


class MLflowModelLoader(ModelLoader):
    def load(
        self,
        model: RegisteredModelVersion,
    ) -> ClassifierMixin:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        model_uri = f"models:/{model.name}/{model.version}"

        return mlflow.sklearn.load_model(model_uri)
