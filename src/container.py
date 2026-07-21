from functools import cached_property
from pathlib import Path

from sklearn.base import ClassifierMixin

from src.dataset import Dataset
from src.embeddings import EmbeddingExtractor
from src.evaluation import Evaluator
from src.inference import Predictor
from src.models import ModelRepository
from src.training import Trainer


class Container:
    def __init__(
        self,
        model_path: Path,
        data_path: Path | None = None,
    ) -> None:

        self._model_path = model_path
        self._data_path = data_path

    @cached_property
    def repository(self) -> ModelRepository:
        return ModelRepository(self._model_path)

    @cached_property
    def dataset(self) -> Dataset:

        if self._data_path is None:
            raise RuntimeError("Dataset path is required.")

        return Dataset(self._data_path)

    @cached_property
    def evaluator(self) -> Evaluator:
        return Evaluator()

    @cached_property
    def extractor(self) -> EmbeddingExtractor:
        return EmbeddingExtractor()

    @cached_property
    def model(self) -> ClassifierMixin:
        return self.repository.load()

    @cached_property
    def trainer(self) -> Trainer:
        return Trainer(
            dataset=self.dataset,
            evaluator=self.evaluator,
            repository=self.repository,
        )

    @cached_property
    def predictor(self) -> Predictor:
        return Predictor(
            model=self.model,
            extractor=self.extractor,
        )
