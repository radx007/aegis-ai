import joblib

from src.config import settings


class ModelRepository:

    def save(
        self,
        model,
    ):

        joblib.dump(
            model,
            settings.baseline_model_path,
        )

        return settings.baseline_model_path

    def load(self):

        return joblib.load(
            settings.baseline_model_path
        )