from functools import cached_property
from pathlib import Path

from sklearn.base import ClassifierMixin

from src.config import settings
from src.dataset import Dataset
from src.embeddings import EmbeddingExtractor
from src.evaluation import Evaluator
from src.exceptions.model_loading import ModelLoadingError
from src.inference import Predictor
from src.mlops.loading.base import ModelLoader
from src.mlops.loading.mlflow_loader import MLflowModelLoader
from src.mlops.metadata import MetadataCollector
from src.mlops.registry import (
    MLflowModelRegistry,
    ModelRegistry,
    NullModelRegistry,
)
from src.mlops.tracking import (
    ExperimentTracker,
    MLflowTracker,
    NullTracker,
)
from src.training import Trainer


class Container:
    def __init__(
        self,
        data_path: Path | None = None,
    ) -> None:

        self._data_path = data_path

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
        if not settings.mlflow_enabled:
            raise ModelLoadingError(
                "MLflow must be enabled to load the production model."
            )

        registered_model = self.registry.get_model_by_alias(
            name=settings.mlflow_model_name,
            alias=settings.mlflow_model_alias,
        )

        return self.loader.load(
            registered_model,
        )

    @cached_property
    def loader(self) -> ModelLoader:
        return MLflowModelLoader()

    @cached_property
    def trainer(self) -> Trainer:
        return Trainer(
            dataset=self.dataset,
            evaluator=self.evaluator,
            tracker=self.tracker,
            metadata_collector=self.metadata_collector,
            registry=self.registry,
        )

    @cached_property
    def predictor(self) -> Predictor:
        return Predictor(
            model=self.model,
            extractor=self.extractor,
        )

    @cached_property
    def tracker(self) -> ExperimentTracker:
        if settings.mlflow_enabled:
            return MLflowTracker()

        return NullTracker()

    @cached_property
    def metadata_collector(self) -> MetadataCollector:
        return MetadataCollector()

    @cached_property
    def registry(self) -> ModelRegistry:
        if settings.mlflow_enabled:
            return MLflowModelRegistry()

        return NullModelRegistry()
