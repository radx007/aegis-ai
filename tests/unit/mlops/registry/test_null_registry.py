import pytest

from src.mlops.registry import NullModelRegistry

pytestmark = pytest.mark.unit


def test_register_model_does_nothing() -> None:
    registry = NullModelRegistry()

    registry.register_model(
        name="aegis-classifier",
    )
