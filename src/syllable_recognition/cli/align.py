"""Target-constrained alignment commands."""

from __future__ import annotations

from pathlib import Path

import click

from .common import echo_json


@click.group("align")
def align_group() -> None:
    """Run target-constrained acoustic alignment."""


@align_group.command("phone-ctc")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--checkpoint", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--audio", "audio_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--text", "target_text", required=True)
def align_phone_ctc_command(
    config_path: Path,
    checkpoint: Path,
    audio_path: Path,
    target_text: str,
) -> None:
    """Align target ARPAbet phones using HuBERT CTC emissions, not MFA."""
    from syllable_recognition.inference.phone_ctc import align_phone_ctc

    echo_json(align_phone_ctc(config_path, checkpoint, audio_path, target_text))
