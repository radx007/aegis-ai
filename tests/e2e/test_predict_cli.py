from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()

pytestmark = pytest.mark.e2e


def test_predict_command(
    e2e_audio: Path,
    champion_model: None,
) -> None:
    # Act
    result = runner.invoke(
        app,
        [
            "predict",
            str(e2e_audio),
        ],
    )

    # Assert
    assert result.exit_code == 0
    assert "(" in result.stdout
    assert "%" in result.stdout