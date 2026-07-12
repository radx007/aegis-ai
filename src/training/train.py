from sklearn.linear_model import LogisticRegression

from src.dataset import Dataset
from src.entities import TrainingResult
from src.evaluation import Evaluator
from src.exceptions import TrainingError
from src.logging import logger
from src.models import ModelRepository


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        evaluator: Evaluator,
        repository: ModelRepository,
    ) -> None:
        self._dataset = dataset
        self._evaluator = evaluator
        self._repository = repository

    def train(self) -> TrainingResult:
        logger.info("Starting model training.")

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self._dataset.split()

        model = LogisticRegression(max_iter=1000)

        try:
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

        except Exception as exc:
            logger.exception("Failed to train model.")

            raise TrainingError("Unable to train model.") from exc

        logger.info("Saving model...")

        model_path = self._repository.save(model)

        logger.success(f"Model saved to {model_path}")

        return TrainingResult(
            metrics=metrics,
            model_path=model_path,
        )
