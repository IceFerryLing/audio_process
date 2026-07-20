"""Model training commands with lazy heavyweight imports."""

from __future__ import annotations

from pathlib import Path

import click

from .common import echo_json, log_stage


@click.group("train")
def train_group() -> None:
    """Train versioned acoustic tasks after their data gates pass."""


@train_group.command("phone-ctc")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--overwrite", is_flag=True, help="Explicitly replace a mismatched correctness run.")
def train_phone_ctc_command(config_path: Path, overwrite: bool) -> None:
    """Run the bounded HuBERT Phone CTC correctness experiment."""
    from syllable_recognition.training.phone_ctc import train_phone_ctc

    log_stage("stage_start", "phone-ctc-correctness")
    report = train_phone_ctc(config_path, overwrite=overwrite)
    log_stage("stage_complete", "phone-ctc-correctness", status=report["status"])
    echo_json(report)
