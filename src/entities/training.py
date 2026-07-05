from dataclasses import dataclass
from pathlib import Path

from .metrics import EvaluationMetrics


@dataclass(slots=True, frozen=True)
class TrainingResult:

    metrics: EvaluationMetrics

    model_path: Path