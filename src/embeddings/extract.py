from pathlib import Path

import librosa
import numpy as np
import tensorflow_hub as hub


class EmbeddingExtractor:

    def __init__(self):

        self.model = hub.load(
            "https://tfhub.dev/google/yamnet/1"
        )

    def extract(
        self,
        audio_path
    ):

        waveform, sr = librosa.load(
            audio_path,
            sr=16000
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