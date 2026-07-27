from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True)
class ExperimentMetadata:
    timestamp: datetime

    python_version: str
    tensorflow_version: str
    numpy_version: str

    operating_system: str
    machine: str
    processor: str

    git_commit: str | None
    git_branch: str | None

    random_seed: int | None
