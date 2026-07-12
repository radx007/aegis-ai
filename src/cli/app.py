import typer

from src.cli.predict import predict_command
from src.cli.train import train_command

app = typer.Typer(help="AEGIS AI command line interface.")

app.command("train")(train_command)
app.command("predict")(predict_command)
