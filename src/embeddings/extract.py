from pathlib import Path
from typing import Any, Protocol

import librosa
import numpy as np
import tensorflow_hub as hub

from src.config import settings
from src.exceptions import EmbeddingError
from src.logging import logger


class _YamnetModel(Protocol):
    def __call__(
        self,
        waveform: np.ndarray,
    ) -> tuple[Any, Any, Any]: ...


class EmbeddingExtractor:
    def __init__(self) -> None:
        try:
            self.model: _YamnetModel = hub.load(settings.yamnet_url)

        except Exception as exc:
            logger.exception("Failed to load embedding model.")

            raise EmbeddingError("Unable to load embedding model.") from exc

    def extract(
        self,
        audio_path: Path,
    ) -> Any:
        try:
            logger.info(f"Extracting embeddings from {audio_path.name}")

            waveform, _sr = librosa.load(
                audio_path,
                sr=settings.sample_rate,
            )

            waveform = waveform.astype(np.float32)

            _scores, embeddings, _spec = self.model(waveform)

            logger.success(f"Extracted embeddings from {audio_path.name}")

            return embeddings.numpy().mean(axis=0)

        except Exception as exc:
            logger.exception(f"Failed to extract embeddings from {audio_path.name}.")

            raise EmbeddingError("Unable to extract embeddings.") from exc
