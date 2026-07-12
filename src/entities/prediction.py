from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PredictionResult:
    label: str

    confidence: float
