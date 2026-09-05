import typer

from src.mlops.bootstrap import configure_mlflow

from .predict import predict_command
from .train import train_command

app = typer.Typer(
    help="AEGIS AI command line interface.",
)


@app.callback()
def _bootstrap() -> None:
    configure_mlflow()


app.command("train")(train_command)
app.command("predict")(predict_command)
