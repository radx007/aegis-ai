from pathlib import Path

import joblib

from src.embeddings import (
    EmbeddingExtractor
)


class Predictor:

    def __init__(self):

        root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        self.model = joblib.load(

            root /

            "models" /

            "baseline.pkl"
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