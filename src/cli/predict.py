from pathlib import Path
from typing import Annotated

import typer

from src.config import settings


def predict_command(
    audio: Path,
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            help="Model path.",
        ),
    ] = settings.baseline_model_path,
) -> None:
    """
    Predict the label of an audio file.
    """
    from src.container import Container

    container = Container(data_path=None, model_path=model)

    result = container.predictor.predict(audio)

    typer.echo(f"{result.label} ({result.confidence:.2%})")
