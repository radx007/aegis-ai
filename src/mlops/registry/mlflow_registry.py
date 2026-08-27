import mlflow

from src.config import settings

from .base import ModelRegistry


class MLflowModelRegistry(ModelRegistry):
    def register_model(
        self,
        name: str,
    ) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        run = mlflow.active_run()

        if run is None:
            raise RuntimeError("An active MLflow run is required to register a model.")

        model_uri = f"runs:/{run.info.run_id}/model"

        mlflow.register_model(
            model_uri=model_uri,
            name=name,
        )

    def promote(
        self,
        name: str,
        version: str,
        alias: str,
    ) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

        client.set_registered_model_alias(
            name=name,
            alias=alias,
            version=version,
        )
