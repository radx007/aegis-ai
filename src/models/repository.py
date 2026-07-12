from pathlib import Path

import joblib
from sklearn.base import ClassifierMixin

from src.config import settings
from src.exceptions import ModelError
from src.logging import logger


class ModelRepository:
    def __init__(self) -> None:
        self._model_path = settings.baseline_model_path

    def save(
        self,
        model: ClassifierMixin,
    ) -> Path:
        try:
            logger.info("Saving model.")

            joblib.dump(
                model,
                self._model_path,
            )

            return self._model_path

        except Exception as exc:
            logger.exception("Failed to save model.")

            raise ModelError("Unable to save model.") from exc

    def load(self) -> ClassifierMixin:
        try:
            logger.info("Loading model.")

            return joblib.load(self._model_path)

        except Exception as exc:
            logger.exception("Failed to load model.")

            raise ModelError("Unable to load model.") from exc
