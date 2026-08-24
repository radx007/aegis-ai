from .base import ModelRegistry


class NullModelRegistry(ModelRegistry):
    def register_model(
        self,
        name: str,
    ) -> None:
        return None
