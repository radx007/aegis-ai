import sys
from pathlib import Path

from loguru import logger

from src.config import settings


def configure_logging() -> None:

    logs_dir: Path = settings.root_path / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level=getattr(settings, "LOG_LEVEL_CONSOLE", "INFO"),
        colorize=True,
        enqueue=True,
    )

    logger.add(
        sink=logs_dir / "aegis.log",
        level=getattr(settings, "LOG_LEVEL_FILE", "DEBUG"),
        rotation=getattr(settings, "LOG_ROTATION", "10 MB"),
        retention=getattr(settings, "LOG_RETENTION", "30 days"),
        compression="zip",
        enqueue=True,
    )


configure_logging()
