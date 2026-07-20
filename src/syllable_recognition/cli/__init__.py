"""Top-level command line interface assembled from domain command groups."""

from __future__ import annotations

import logging

import click

from .align import align_group
from .data import data_group
from .evaluate import evaluate_group
from .train import train_group


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@click.group()
def main() -> None:
    """Syllable recognition data, training, evaluation, and inference tools."""
    _configure_logging()


main.add_command(data_group)
main.add_command(train_group)
main.add_command(align_group)
main.add_command(evaluate_group)


__all__ = ["main"]


if __name__ == "__main__":
    main()
