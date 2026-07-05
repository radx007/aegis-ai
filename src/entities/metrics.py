from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class EvaluationMetrics:

    accuracy: float

    precision: float

    recall: float

    f1: float

    confusion_matrix: np.ndarray