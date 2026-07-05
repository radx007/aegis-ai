from sklearn.linear_model import (
    LogisticRegression
)

from src.config import settings

from src.dataset import Dataset

from src.evaluation import Evaluator

from src.models import ModelRepository

class Trainer:

    def __init__(self):

        self.data = (
            settings.processed_data_path
        )

        self.models = (
            settings.models_path
        )

        self.models.mkdir(
            exist_ok=True
        )

    def train(self):

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

        pred = model.predict(
            X_test
        )

        evaluator = Evaluator()

        metrics = evaluator.evaluate(
            y_test,
            pred,
        )

        repository = ModelRepository()

        path = repository.save(
            model
        )

        return {

            **metrics,
            "path": str(
                path
            )
        }