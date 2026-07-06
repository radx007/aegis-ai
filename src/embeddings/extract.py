from pathlib import Path
from typing import Protocol

import librosa
import numpy as np
import tensorflow_hub as hub

from src.config import settings

from src.logging import logger

class _YamnetModel(Protocol):

    def __call__(
        self,
        waveform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class EmbeddingExtractor:

    def __init__(self) -> None:

        self.model: _YamnetModel = hub.load(
            settings.yamnet_url
        )

    def extract(
        self,
        audio_path: Path,
    ) -> np.ndarray:
        
        logger.info(
            f"Extracting embeddings from {audio_path.name}"
        )

        waveform, _sr = librosa.load(
            audio_path,
            sr=settings.sample_rate,
        )

        waveform = waveform.astype(
            np.float32
        )

        _scores, \
        embeddings, \
        _spec = self.model(
            waveform
        )

        return (
            embeddings
            .numpy()
            .mean(axis=0)
        )