"""Command-line entrypoints for the land classifier project."""

from land_classifier.training.train import main as train


def main() -> None:
    train()


if __name__ == "__main__":
    main()
