
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from src.embeddings import (
    EmbeddingExtractor
)

from src.entities import PredictionResult
from src.models import ModelRepository

from pathlib import Path

from src.logging import logger

class Predictor:

    def __init__(self) -> None:

        repository = ModelRepository()

        self.model = repository.load()

        self.extractor: EmbeddingExtractor = (
            EmbeddingExtractor()
        )

    def predict(
        self,
        audio_path: Path,
    ) -> PredictionResult:
        
        logger.info(
            f"Predicting {audio_path.name}"
        )

        emb: np.ndarray = (
            self.extractor
            .extract(
                audio_path
            )
        )

        probs: np.ndarray = (
            self.model
            .predict_proba(
                np.expand_dims(
                    emb,
                    axis=0,
                )
            )[0]
        )

        label = str(
            self.model
            .classes_[
                int(probs.argmax())
            ]
        )

        confidence: float = (
            probs.max()
        )

        logger.success(
            f"{label} ({confidence:.2%})"
        )

        return PredictionResult(
            label=label,
            confidence=float(confidence),
        )