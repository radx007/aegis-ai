from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from typer.testing import CliRunner

from src.cli.app import app
from src.config import settings

runner = CliRunner()

pytestmark = pytest.mark.e2e


def test_train_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve().as_posix()}"

    monkeypatch.setattr(
        settings,
        "mlflow_tracking_uri",
        tracking_uri,
    )

    X, y = make_classification(
        n_samples=50,
        n_features=1024,
        n_informative=20,
        n_redundant=0,
        random_state=42,
    )

    processed = tmp_path / "processed"
    processed.mkdir()

    np.save(
        processed / "X.npy",
        X,
    )

    np.save(
        processed / "y.npy",
        y,
    )

    # Act
    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(processed),
        ],
    )

    # Assert
    assert result.exit_code == 0
    assert "Training completed." in result.stdout
