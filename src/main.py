from src.container import Container


def main() -> None:
    container = Container()

    trainer = container.trainer

    predictor = container.predictor

    # CLI / API / experiments will use these objects


if __name__ == "__main__":
    main()