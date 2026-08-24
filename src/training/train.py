from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.dataset import Dataset
from src.entities import TrainingResult
from src.evaluation import Evaluator
from src.exceptions import TrainingError
from src.logging import logger
from src.mlops.metadata import MetadataCollector
from src.mlops.tracking import ExperimentTracker
from src.models import ModelRepository


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        evaluator: Evaluator,
        repository: ModelRepository,
        tracker: ExperimentTracker,
        metadata_collector: MetadataCollector,
    ) -> None:
        self._dataset = dataset
        self._evaluator = evaluator
        self._repository = repository
        self._tracker = tracker
        self._metadata_collector = metadata_collector

    def train(self) -> TrainingResult:
        logger.info("Starting model training.")

        self._tracker.start_run("baseline")

        metadata = self._metadata_collector.collect()

        self._tracker.log_metadata(metadata)

        self._tracker.log_parameters(
            {
                "max_iter": float(settings.training_max_iter),
                "random_state": float(settings.training_random_state),
            }
        )

        try:
            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = self._dataset.split()

            model = LogisticRegression(
                max_iter=settings.training_max_iter,
                random_state=settings.training_random_state,
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

            logger.info("Saving model...")

            model_path = self._repository.save(model)

            self._tracker.log_model(model)

            logger.success(f"Model saved to {model_path}")

            return TrainingResult(
                metrics=metrics,
                model_path=model_path,
            )

        except Exception as exc:
            logger.exception("Failed to train model.")

            raise TrainingError("Unable to train model.") from exc

        finally:
            self._tracker.end_run()
