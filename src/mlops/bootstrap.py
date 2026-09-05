import mlflow

from src.config import settings


def configure_mlflow() -> None:
    if not settings.mlflow_enabled:
        return

    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri,
    )
