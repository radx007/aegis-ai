from pathlib import Path

import numpy as np
from sklearn.base import ClassifierMixin

from src.embeddings import EmbeddingExtractor
from src.entities import PredictionResult
from src.exceptions import PredictionError
from src.logging import logger
from src.models import ModelRepository


class Predictor:

    def __init__(
        self,
        model: ClassifierMixin,
        extractor: EmbeddingExtractor,
    ) -> None:
        self._extractor = extractor
        self._model = model

    def predict(
        self,
        audio_path: Path,
    ) -> PredictionResult:

        logger.info(f"Predicting {audio_path.name}")

        embedding = self._extractor.extract(audio_path)

        try:
            probabilities = self._model.predict_proba(
                np.expand_dims(embedding, axis=0)
            )[0]

            label = str(
                self._model.classes_[
                    int(probabilities.argmax())
                ]
            )

            confidence = float(probabilities.max())

        except Exception as exc:

            logger.exception(
                f"Failed to predict {audio_path.name}."
            )

            raise PredictionError(
                "Unable to make prediction."
            ) from exc

        logger.success(
            f"{label} ({confidence:.2%})"
        )

        return PredictionResult(
            label=label,
            confidence=confidence,
        )