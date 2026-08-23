from pathlib import Path

import pytest

from src.mlops.registry import NullModelRegistry

pytestmark = pytest.mark.unit


def test_register_model_does_nothing() -> None:
    registry = NullModelRegistry()

    model_path = Path("baseline.pkl")

    registry.register_model(
        model=model_path,
        name="aegis-classifier",
    )
