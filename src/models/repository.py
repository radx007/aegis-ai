from pathlib import Path

import joblib

from src.config import settings

from sklearn.base import ClassifierMixin

from src.logging import logger

class ModelRepository:

    def save(
        self,
        model: ClassifierMixin,
    )-> Path:
        
        logger.info("Saving model.")

        joblib.dump(
            model,
            settings.baseline_model_path,
        )

        return settings.baseline_model_path

    def load(self) -> ClassifierMixin:

        logger.info("Loading model.")

        return joblib.load(
            settings.baseline_model_path
        )