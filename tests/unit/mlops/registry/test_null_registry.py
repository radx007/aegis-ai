import pytest

from src.config import settings
from src.mlops.registry import NullModelRegistry

pytestmark = pytest.mark.unit


def test_register_model_does_nothing() -> None:
    registry = NullModelRegistry()

    registry.register_model(
        name=settings.mlflow_model_name,
    )


def test_promote_does_nothing() -> None:
    registry = NullModelRegistry()

    registry.promote(
        name=settings.mlflow_model_name,
        version="8",
        alias=settings.mlflow_model_alias,
    )


def test_get_model_by_alias_returns_empty_version() -> None:
    registry = NullModelRegistry()

    result = registry.get_model_by_alias(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
    )

    assert result.name == settings.mlflow_model_name
    assert result.version == ""
