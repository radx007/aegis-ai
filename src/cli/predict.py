from pathlib import Path

import typer

def predict_command(
    audio: Path,
) -> None:
    """
    Predict the label of an audio file.
    """
    from src.container import Container

    container = Container()

    result = (
        container.predictor.predict(audio)
    )

    typer.echo(
        f"{result.label} "
        f"({result.confidence:.2%})"
    )