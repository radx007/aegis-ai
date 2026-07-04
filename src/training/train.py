from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score
)

from src.config import settings

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

    def load_data(self):

        X = np.load(
            self.data /
            "X.npy"
        )

        y = np.load(
            self.data /
            "y.npy"
        )

        return X, y

    def train(self):

        X, y = self.load_data()

        (
            X_train,
            X_test,
            y_train,
            y_test

        ) = train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42
        )

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

        acc = accuracy_score(
            y_test,
            pred
        )

        path = (
            settings.baseline_model_path
        )

        joblib.dump(
            model,
            path
        )

        return {

            "accuracy": acc,

            "path": str(
                path
            )
        }