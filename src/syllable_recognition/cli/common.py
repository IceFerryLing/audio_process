"""Small presentation helpers shared by CLI command modules."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import click


LOGGER = logging.getLogger("sylrec")


def echo_json(payload: Mapping[str, Any]) -> None:
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


def log_stage(event: str, stage: str, **fields: Any) -> None:
    LOGGER.info(json.dumps({"event": event, "stage": stage, **fields}, sort_keys=True))
