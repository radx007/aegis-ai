from dataclasses import dataclass

from .metrics import EvaluationMetrics


@dataclass(slots=True, frozen=True)
class TrainingResult:
    metrics: EvaluationMetrics
