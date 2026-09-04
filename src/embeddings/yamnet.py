from pathlib import Path
from typing import Any, Protocol

import librosa
import numpy as np
import tensorflow_hub as hub

from src.exceptions import EmbeddingError
from src.logging import logger

from .base import EmbeddingModel


class _YamnetModel(Protocol):
    def __call__(
        self,
        waveform: np.ndarray,
    ) -> tuple[Any, Any, Any]: ...


class YamnetEmbeddingExtractor(EmbeddingModel):
    def __init__(
        self,
        model_url: str,
        sample_rate: int,
    ) -> None:
        self._sample_rate = sample_rate

        try:
            self._model: _YamnetModel = hub.load(
                model_url,
            )
        except Exception as exc:
            logger.exception("Failed to load embedding model.")

            raise EmbeddingError("Unable to load embedding model.") from exc

    def extract(
        self,
        audio_path: Path,
    ) -> np.ndarray:
        try:
            logger.info(f"Extracting embeddings from {audio_path.name}")

            waveform, _sr = librosa.load(
                audio_path,
                sr=self._sample_rate,
            )

            waveform = waveform.astype(
                np.float32,
            )

            _scores, embeddings, _spec = self._model(
                waveform,
            )

            embedding: np.ndarray = np.asarray(
                embeddings.numpy().mean(
                    axis=0,
                ),
                dtype=np.float32,
            )

            logger.success(f"Extracted embeddings from {audio_path.name}")

            return embedding

        except Exception as exc:
            logger.exception(f"Failed to extract embeddings from {audio_path.name}.")

            raise EmbeddingError("Unable to extract embeddings.") from exc
