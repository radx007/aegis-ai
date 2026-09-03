from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run_name: str
    max_iter: int
    random_state: int
    model_name: str
    model_alias: str