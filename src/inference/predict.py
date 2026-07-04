from pathlib import Path

import joblib

from src.config import settings
from src.embeddings import (
    EmbeddingExtractor
)


class Predictor:

    def __init__(self):

        self.model = joblib.load(
            settings.baseline_model_path
        )

        self.extractor = (
            EmbeddingExtractor()
        )

    def predict(
        self,
        audio_path
    ):

        emb = (
            self.extractor
            .extract(
                audio_path
            )
        )

        probs = (
            self.model
            .predict_proba(
                [emb]
            )[0]
        )

        label = (
            self.model
            .classes_[
                probs.argmax()
            ]
        )

        confidence = (
            probs.max()
        )

        return {

            "label": label,

            "confidence": float(
                confidence
            )
        }