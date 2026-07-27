from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.entities.experiment_metadata import ExperimentMetadata


class MetadataCollector:
    """
    Collects metadata describing the current experiment environment.
    """

    def collect(
        self,
        *,
        random_seed: int | None = None,
    ) -> ExperimentMetadata:
        return ExperimentMetadata(
            timestamp=datetime.now(),
            python_version=sys.version.split()[0],
            tensorflow_version=tf.__version__,
            numpy_version=np.__version__,
            operating_system=platform.system(),
            machine=platform.machine(),
            processor=platform.processor(),
            git_commit=self._git_commit(),
            git_branch=self._git_branch(),
            random_seed=random_seed,
        )

    @staticmethod
    def _git_commit() -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            return None

    @staticmethod
    def _git_branch() -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            return None
