from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.embeddings import YamnetEmbeddingExtractor
from src.exceptions import EmbeddingError

pytestmark = pytest.mark.unit


@pytest.fixture
def model_url() -> str:
    return "fake-yamnet-url"


@pytest.fixture
def sample_rate() -> int:
    return 16000


def test_constructor_loads_model(
    model_url: str,
    sample_rate: int,
) -> None:
    with patch(
        "src.embeddings.yamnet.hub.load",
        return_value=Mock(),
    ) as mock_load:
        YamnetEmbeddingExtractor(
            model_url=model_url,
            sample_rate=sample_rate,
        )

    mock_load.assert_called_once_with(
        model_url,
    )


def test_extract_returns_embedding(
    mock_tfhub_model: Mock,
    fake_audio: tuple[np.ndarray, int],
    model_url: str,
    sample_rate: int,
) -> None:
    with patch(
        "src.embeddings.yamnet.hub.load",
        return_value=mock_tfhub_model,
    ):
        extractor = YamnetEmbeddingExtractor(
            model_url=model_url,
            sample_rate=sample_rate,
        )

    with patch(
        "src.embeddings.yamnet.librosa.load",
        return_value=fake_audio,
    ) as mock_load:
        embedding = extractor.extract(
            Path("audio.wav"),
        )

    mock_load.assert_called_once_with(
        Path("audio.wav"),
        sr=sample_rate,
    )

    assert embedding.shape == (1024,)


def test_constructor_raises_embedding_error(
    model_url: str,
    sample_rate: int,
) -> None:
    with patch(
        "src.embeddings.yamnet.hub.load",
        side_effect=Exception,
    ):
        with pytest.raises(EmbeddingError):
            YamnetEmbeddingExtractor(
                model_url=model_url,
                sample_rate=sample_rate,
            )


def test_extract_raises_embedding_error(
    model_url: str,
    sample_rate: int,
) -> None:
    with patch(
        "src.embeddings.yamnet.hub.load",
        return_value=Mock(),
    ):
        extractor = YamnetEmbeddingExtractor(
            model_url=model_url,
            sample_rate=sample_rate,
        )

    with patch(
        "src.embeddings.yamnet.librosa.load",
        side_effect=Exception,
    ):
        with pytest.raises(EmbeddingError):
            extractor.extract(
                Path("audio.wav"),
            )
