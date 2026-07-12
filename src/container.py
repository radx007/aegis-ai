from functools import cached_property

from src.dataset import Dataset
from src.embeddings import EmbeddingExtractor
from src.evaluation import Evaluator
from src.inference import Predictor
from src.models import ModelRepository
from src.training import Trainer


class Container:
    """Application composition root managing component lifecycles.
    Uses thread-safe cached singletons.
    """

    def __init__(self) -> None:
        self._repository = ModelRepository()
        self._dataset = Dataset()
        self._evaluator = Evaluator()
        self._extractor = EmbeddingExtractor()

    @cached_property
    def model(self):
        """Lazily loads the heavy machine learning model on first access."""
        return self._repository.load()

    @cached_property
    def trainer(self) -> Trainer:
        """Returns a cached, single instance of the Trainer."""
        return Trainer(
            dataset=self._dataset,
            evaluator=self._evaluator,
            repository=self._repository,
        )

    @cached_property
    def predictor(self) -> Predictor:
        """Returns a cached, single instance of the Predictor."""
        return Predictor(
            model=self.model,
            extractor=self._extractor,
        )
