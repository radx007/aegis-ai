from pathlib import Path
from unittest.mock import Mock

import pytest

from src.entities import PredictionResult
from src.exceptions.prediction import PredictionError
from src.inference import Predictor

pytestmark = pytest.mark.unit


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


def test_predict_raises_prediction_error_when_model_fails(
    mock_model_failure: Mock,
    mock_extractor: Mock,
) -> None:

    predictor = Predictor(
        model=mock_model_failure,
        extractor=mock_extractor,
    )

    with pytest.raises(PredictionError):
        predictor.predict(Path("audio.wav"))


def test_predict_raises_prediction_error_when_extractor_fails(
    mock_model: Mock,
    mock_extractor_failure: Mock,
) -> None:

    predictor = Predictor(
        model=mock_model,
        extractor=mock_extractor_failure,
    )

    with pytest.raises(PredictionError):
        predictor.predict(Path("audio.wav"))

    mock_model.predict_proba.assert_not_called()
