from pathlib import Path

from pydantic_settings import BaseSettings

_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    sample_rate: int = 16000

    root_path: Path = _ROOT

    data_path: Path = _ROOT / "data"

    models_path: Path = _ROOT / "models"

    processed_data_path: Path = _ROOT / "data" / "processed"

    baseline_model_path: Path = _ROOT / "models" / "baseline.pkl"

    yamnet_url: str = "https://tfhub.dev/google/yamnet/1"

    class Config:
        env_file = ".env"


settings = Settings()
