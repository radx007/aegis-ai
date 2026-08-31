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
) -> None:
    """
    Train the classifier.
    """
    from src.container import Container

    container = Container(data_path=data)

    container.trainer.train()

    typer.echo("Training completed.")
