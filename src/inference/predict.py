
from src.embeddings import (
    EmbeddingExtractor
)
from src.entities import PredictionResult
from src.models import ModelRepository


class Predictor:

    def __init__(self):

        repository = ModelRepository()

        self.model = repository.load()

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

        return PredictionResult(
            label=label,
            confidence=float(confidence),
        )