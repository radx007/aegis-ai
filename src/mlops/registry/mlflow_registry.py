import mlflow

from src.entities import RegisteredModelVersion

from .base import ModelRegistry


class MLflowModelRegistry(ModelRegistry):
    def register_model(
        self,
        name: str,
    ) -> RegisteredModelVersion:
        run = mlflow.active_run()

        if run is None:
            raise RuntimeError("An active MLflow run is required to register a model.")

        model_uri = f"runs:/{run.info.run_id}/model"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=name,
        )

        return RegisteredModelVersion(
            name=model_version.name,
            version=model_version.version,
        )

    def promote(
        self,
        name: str,
        version: str,
        alias: str,
    ) -> None:
        client = mlflow.MlflowClient()

        client.set_registered_model_alias(
            name=name,
            alias=alias,
            version=version,
        )

    def get_model_by_alias(
        self,
        name: str,
        alias: str,
    ) -> RegisteredModelVersion:
        client = mlflow.MlflowClient()

        model_version = client.get_model_version_by_alias(
            name=name,
            alias=alias,
        )

        return RegisteredModelVersion(
            name=model_version.name,
            version=model_version.version,
        )
