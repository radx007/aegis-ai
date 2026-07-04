from pathlib import Path

import librosa
import numpy as np
import tensorflow_hub as hub

from src.config import settings


class EmbeddingExtractor:

    def __init__(self):

        self.model = hub.load(
            settings.yamnet_url
        )

    def extract(
        self,
        audio_path
    ):

        waveform, sr = librosa.load(
            audio_path,
            sr= settings.sample_rate
        )

        waveform = waveform.astype(
            np.float32
        )

        scores,\
        embeddings,\
        spec = self.model(
            waveform
        )

        return (
            embeddings
            .numpy()
            .mean(axis=0)
        )