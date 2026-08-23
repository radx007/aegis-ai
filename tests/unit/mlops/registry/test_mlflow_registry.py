from pathlib import Path

import pytest

from src.mlops.registry import MLflowModelRegistry

pytestmark = pytest.mark.unit


def test_register_model_raises_not_implemented_error() -> None:
    registry = MLflowModelRegistry()

    with pytest.raises(NotImplementedError):
        registry.register_model(
            model=Path("baseline.pkl"),
            name="aegis-classifier",
        )
