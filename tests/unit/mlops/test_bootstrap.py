from unittest.mock import patch

import pytest

from src.mlops.bootstrap import configure_mlflow

pytestmark = pytest.mark.unit


def test_configures_mlflow_when_enabled() -> None:
    with (
        patch(
            "src.mlops.bootstrap.settings.mlflow_enabled",
            True,
        ),
        patch(
            "src.mlops.bootstrap.mlflow.set_tracking_uri",
        ) as set_tracking_uri,
    ):
        configure_mlflow()

    set_tracking_uri.assert_called_once()


def test_does_not_configure_mlflow_when_disabled() -> None:
    with (
        patch(
            "src.mlops.bootstrap.settings.mlflow_enabled",
            False,
        ),
        patch(
            "src.mlops.bootstrap.mlflow.set_tracking_uri",
        ) as set_tracking_uri,
    ):
        configure_mlflow()

    set_tracking_uri.assert_not_called()
