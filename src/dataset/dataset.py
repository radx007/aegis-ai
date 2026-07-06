from sklearn.model_selection import train_test_split

import numpy as np

from src.config import settings

from src.logging import logger


class Dataset:

    def load(self) -> tuple[np.ndarray, np.ndarray]:

        X = np.load(
            settings.processed_data_path / "X.npy"
        )

        y = np.load(
            settings.processed_data_path / "y.npy"
        )

        return X, y

    def split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        logger.info(
            "Loading processed dataset."
        )

        X, y = self.load()

        logger.success(
            f"Loaded {len(X)} samples."
        )

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )