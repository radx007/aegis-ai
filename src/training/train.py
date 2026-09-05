from sklearn.linear_model import LogisticRegression

from src.dataset import Dataset
from src.entities import TrainingResult
from src.evaluation import Evaluator
from src.exceptions import TrainingError
from src.logging import logger
from src.mlops.metadata import MetadataCollector
from src.mlops.registry import ModelRegistry
from src.mlops.tracking import ExperimentTracker
from src.training import TrainingConfig


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        evaluator: Evaluator,
        tracker: ExperimentTracker,
        metadata_collector: MetadataCollector,
        registry: ModelRegistry,
        config: TrainingConfig,
    ) -> None:
        self._dataset = dataset
        self._evaluator = evaluator
        self._tracker = tracker
        self._metadata_collector = metadata_collector
        self._registry = registry
        self._config = config

    def train(self) -> TrainingResult:
        logger.info("Starting model training.")

        self._tracker.start_run(self._config.run_name)

        metadata = self._metadata_collector.collect()

        self._tracker.log_metadata(metadata)

        self._tracker.log_parameters(
            {
                "max_iter": float(self._config.max_iter),
                "random_state": float(self._config.random_state),
            }
        )

        try:
            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = self._dataset.split(
                test_size=self._config.test_size,
                random_state=self._config.random_state,
            )

            model = LogisticRegression(
                max_iter=self._config.max_iter,
                random_state=self._config.random_state,
            )

            model.fit(
                X_train,
                y_train,
            )

            logger.success("Model trained successfully.")

            predictions = model.predict(X_test)

            metrics = self._evaluator.evaluate(
                y_test,
                predictions,
            )

            self._tracker.log_metrics(
                {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )

            logger.info("Logging model to MLflow...")

            self._tracker.log_model(model)

            registered_model = self._registry.register_model(
                name=self._config.model_name,
            )

            self._registry.promote(
                name=registered_model.name,
                version=registered_model.version,
                alias=self._config.model_alias,
            )

            logger.success("Model logged, registered, and promoted successfully.")

            return TrainingResult(
                metrics=metrics,
            )

        except Exception as exc:
            logger.exception("Failed to train model.")

            raise TrainingError("Unable to train model.") from exc

        finally:
            self._tracker.end_run()
