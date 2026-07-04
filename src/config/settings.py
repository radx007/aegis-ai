from pathlib import Path

from pydantic_settings import (
    BaseSettings
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


class Settings(
    BaseSettings
):

    sample_rate: int = 16000

    data_path: Path = ROOT / "data"

    models_path: Path = ROOT / "models"

    processed_data_path: Path = ROOT / "data" / "processed"

    baseline_model_path: Path = (
        ROOT / "models" / "baseline.pkl"
    )

    yamnet_url: str = (
        "https://tfhub.dev/google/yamnet/1"
    )

    class Config:

        env_file = ".env"


settings = Settings()