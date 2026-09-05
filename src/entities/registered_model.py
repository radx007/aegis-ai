from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RegisteredModelVersion:
    name: str

    version: str
