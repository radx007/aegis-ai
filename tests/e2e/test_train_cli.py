from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from typer.testing import CliRunner

from src.cli.app import app

runner = CliRunner()

pytestmark = pytest.mark.E2E


def test_train_command(
    tmp_path: Path,
) -> None:
    # Arrange
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

    model_path = tmp_path / "baseline.pkl"

    # Act
    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(processed),
            "--model",
            str(model_path),
        ],
    )

    # Assert
    assert result.exit_code == 0

    assert model_path.exists()

    assert "Training completed." in result.stdout

    assert str(model_path) in result.stdout
