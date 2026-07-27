import pytest

from src.entities.experiment_metadata import ExperimentMetadata
from src.mlops.metadata import MetadataCollector

pytestmark = pytest.mark.unit


def test_collect_returns_experiment_metadata() -> None:
    collector = MetadataCollector()

    metadata = collector.collect()

    assert isinstance(metadata, ExperimentMetadata)

    assert metadata.python_version
    assert metadata.tensorflow_version
    assert metadata.numpy_version

    assert metadata.operating_system
    assert metadata.machine

    assert metadata.timestamp is not None


def test_collect_stores_random_seed() -> None:
    collector = MetadataCollector()

    metadata = collector.collect(
        random_seed=42,
    )

    assert metadata.random_seed == 42
