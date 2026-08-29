from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    sample_rate: int = 16000

    root_path: Path = _ROOT

    data_path: Path = _ROOT / "data"

    models_path: Path = _ROOT / "models"

    processed_data_path: Path = _ROOT / "data" / "processed"

    baseline_model_path: Path = _ROOT / "models" / "baseline.pkl"

    yamnet_url: str = "https://tfhub.dev/google/yamnet/1"

    training_max_iter: int = 1000

    training_random_state: int = 42

    mlflow_tracking_uri: str = f"sqlite:///{(_ROOT / 'mlflow.db').resolve().as_posix()}"

    mlflow_experiment_name: str = "aegis-ai"

    mlflow_model_name: str = "aegis-classifier"

    mlflow_model_alias: str = "champion"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
