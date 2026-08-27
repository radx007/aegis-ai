import pytest

from src.mlops.registry import NullModelRegistry

pytestmark = pytest.mark.unit


def test_register_model_does_nothing() -> None:
    registry = NullModelRegistry()

    registry.register_model(
        name="aegis-classifier",
    )


def test_promote_does_nothing() -> None:
    registry = NullModelRegistry()

    registry.promote(
        name="aegis-classifier",
        version="8",
        alias="champion",
    )


def test_get_model_by_alias_returns_empty_version() -> None:
    registry = NullModelRegistry()

    result = registry.get_model_by_alias(
        name="aegis-classifier",
        alias="champion",
    )

    assert result.name == "aegis-classifier"
    assert result.version == ""
