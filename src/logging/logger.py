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
        level=settings.log_level_console,
        colorize=True,
        enqueue=True,
    )

    logger.add(
        sink=logs_dir / "aegis.log",
        level=settings.log_level_file,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
        enqueue=True,
    )


configure_logging()
