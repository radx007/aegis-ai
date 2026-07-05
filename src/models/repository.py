from pathlib import Path

import joblib

from src.config import settings

from sklearn.base import ClassifierMixin



class ModelRepository:

    def save(
        self,
        model: ClassifierMixin,
    )-> Path:

        joblib.dump(
            model,
            settings.baseline_model_path,
        )

        return settings.baseline_model_path

    def load(self) -> ClassifierMixin:

        return joblib.load(
            settings.baseline_model_path
        )