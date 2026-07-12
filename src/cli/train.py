import typer

def train_command() -> None:
    """
    Train the classifier.
    """
    from src.container import Container 

    container = Container()

    result = container.trainer.train()

    typer.echo(
        f"Training completed.\n"
        f"Model: {result.model_path}"
    )