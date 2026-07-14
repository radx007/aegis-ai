from pathlib import Path
from unittest.mock import Mock

from src.entities import PredictionResult
from src.inference import Predictor


def test_predict_returns_prediction_result(
    mock_model: Mock,
    mock_extractor: Mock,
) -> None:

    predictor = Predictor(
        model=mock_model,
        extractor=mock_extractor,
    )

    result = predictor.predict(Path("audio.wav"))

    assert isinstance(
        result,
        PredictionResult,
    )

    assert result.label == "siren"

    assert result.confidence == 0.95
