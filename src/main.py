import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from src.cli.app import app  # noqa: E402


def main() -> None:
    app()


if __name__ == "__main__":
    main()
