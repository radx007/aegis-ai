from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import settings
from src.exceptions import DatasetError
from src.logging import logger


class Dataset:
    def load(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            X = np.load(settings.processed_data_path / "X.npy")

            y = np.load(settings.processed_data_path / "y.npy")

            return X, y

        except Exception as exc:
            logger.exception("Failed to load processed dataset.")

            raise DatasetError("Unable to load processed dataset.") from exc

    def split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            logger.info("Loading processed dataset.")

            X, y = self.load()

            logger.success(f"Loaded {len(X)} samples.")

            split = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )

            return cast(
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
                split,
            )

        except Exception as exc:
            logger.exception("Failed to split dataset.")

            raise DatasetError("Unable to split processed dataset.") from exc
