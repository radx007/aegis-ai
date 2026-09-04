import typer

from src.cli.predict import predict_command
from src.cli.train import train_command
from src.mlops.bootstrap import configure_mlflow

app = typer.Typer(
    help="AEGIS AI command line interface.",
)


@app.callback()
def _bootstrap() -> None:
    configure_mlflow()


app.command("train")(train_command)
app.command("predict")(predict_command)
