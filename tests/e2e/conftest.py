from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from typer.testing import CliRunner

from src.cli.app import app
from src.config import settings
from src.mlops.registry import MLflowModelRegistry


@pytest.fixture
def e2e_audio(tmp_path: Path) -> Path:
    """
    Return a real WAV fixture used by prediction E2E tests.
    """
    source = Path("tests/e2e/fixtures/audio/test.wav")

    if not source.exists():
        pytest.fail(f"Missing E2E audio fixture: {source}")

    return source


@pytest.fixture
def champion_model(tmp_path: Path) -> None:
    """
    Prepare a real MLflow champion model for E2E tests.
    """
    X, y = make_classification(
        n_samples=50,
        n_features=1024,
        n_informative=20,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    processed = tmp_path / "processed"
    processed.mkdir()

    np.save(processed / "X.npy", X)
    np.save(processed / "y.npy", y)

    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(processed),
        ],
    )

    assert result.exit_code == 0
    assert "Training completed." in result.stdout

    registry = MLflowModelRegistry()

    registered_model = registry.get_model_by_alias(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
    )

    assert registered_model.name == settings.mlflow_model_name
    assert registered_model.version
