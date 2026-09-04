from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingModel(ABC):
    @abstractmethod
    def extract(
        self,
        audio_path: Path,
    ) -> np.ndarray: ...
