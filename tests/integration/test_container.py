import pytest

from src.container import Container
from src.inference import Predictor
from src.training import Trainer

pytestmark = pytest.mark.integration


def test_container_builds_trainer() -> None:

    container = Container()

    trainer = container.trainer

    assert isinstance(
        trainer,
        Trainer,
    )

    assert trainer._dataset is not None
    assert trainer._repository is not None
    assert trainer._evaluator is not None


def test_container_builds_predictor() -> None:

    container = Container()

    predictor = container.predictor

    assert isinstance(
        predictor,
        Predictor,
    )

    assert predictor._model is not None
    assert predictor._extractor is not None
