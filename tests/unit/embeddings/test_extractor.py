from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.embeddings import EmbeddingExtractor
from src.exceptions.embedding import EmbeddingError


def test_constructor_loads_model() -> None:

    with patch(
        "src.embeddings.extract.hub.load",
        return_value=Mock(),
    ) as mock_load:
        EmbeddingExtractor()

    mock_load.assert_called_once()


def test_extract_returns_embedding(
    mock_tfhub_model: Mock, fake_audio: tuple[np.ndarray, int]
) -> None:

    with patch(
        "src.embeddings.extract.hub.load",
        return_value=mock_tfhub_model,
    ):
        extractor = EmbeddingExtractor()

    with patch(
        "src.embeddings.extract.librosa.load",
        return_value=(
            fake_audio[0],
            fake_audio[1],
        ),
    ):
        embedding = extractor.extract(Path("audio.wav"))

    assert embedding.shape == (1024,)


def test_constructor_raises_embedding_error() -> None:

    with patch(
        "src.embeddings.extract.hub.load",
        side_effect=Exception,
    ):
        with pytest.raises(EmbeddingError):
            EmbeddingExtractor()


def test_extract_raises_embedding_error() -> None:

    with patch(
        "src.embeddings.extract.hub.load",
        return_value=Mock(),
    ):
        extractor = EmbeddingExtractor()

    with patch(
        "src.embeddings.extract.librosa.load",
        side_effect=Exception,
    ):
        with pytest.raises(EmbeddingError):
            extractor.extract(Path("audio.wav"))
