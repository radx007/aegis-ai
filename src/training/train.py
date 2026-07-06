from pathlib import Path

from sklearn.linear_model import (
    LogisticRegression
)

from src.config import settings

from src.dataset import Dataset

from src.entities import TrainingResult

from src.evaluation import Evaluator

from src.models import ModelRepository

from src.logging import logger

class Trainer:

    def __init__(self) -> None:

        self.data: Path = (
            settings.processed_data_path
        )

        self.models: Path = (
            settings.models_path
        )

        self.models.mkdir(
            exist_ok=True
        )

    def train(self) -> TrainingResult:

        logger.info("Starting model training.")

        dataset = Dataset()

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = dataset.split()

        model = (
            LogisticRegression(
                max_iter=1000
            )
        )

        model.fit(
            X_train,
            y_train
        )

        logger.success("Model trained successfully.")

        pred = model.predict(
            X_test
        )

        evaluator = Evaluator()

        metrics = evaluator.evaluate(
            y_test,
            pred,
        )

        repository = ModelRepository()

        logger.info("Saving model...")

        path = repository.save(
            model
        )

        logger.success(
            f"Model saved to {path}"
        )

        return TrainingResult(
            metrics=metrics,
            model_path=path,
        )