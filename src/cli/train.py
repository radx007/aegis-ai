from pathlib import Path
from typing import Annotated

import typer

from src.config import settings


def train_command(
    data: Annotated[
        Path,
        typer.Option(
            "--data",
            help="Processed dataset directory.",
        ),
    ] = settings.processed_data_path,
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            help="Output model path.",
        ),
    ] = settings.baseline_model_path,
) -> None:
    """
    Train the classifier.
    """
    from src.container import Container

    container = Container(data_path=data, model_path=model)

    result = container.trainer.train()

    typer.echo(f"Training completed.\nModel: {result.model_path}")
