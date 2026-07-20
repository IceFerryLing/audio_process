"""Read-only evaluation commands."""

from __future__ import annotations

from pathlib import Path

import click

from syllable_recognition.core.artifacts import write_json

from .common import echo_json


@click.group("evaluate")
def evaluate_group() -> None:
    """Evaluate checkpoints without changing training state."""


@evaluate_group.command("phone-ctc-mfa")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--checkpoint", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--phone-manifest", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--mfa-manifest", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--item-id", required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def evaluate_phone_ctc_mfa_command(
    config_path: Path,
    checkpoint: Path,
    phone_manifest: Path,
    mfa_manifest: Path,
    item_id: str,
    output: Path,
) -> None:
    """Compare CTC emission boundaries with an MFA reference item."""
    from syllable_recognition.evaluation.phone_ctc_mfa import evaluate_phone_ctc_against_mfa

    report = evaluate_phone_ctc_against_mfa(
        training_config=config_path,
        checkpoint=checkpoint,
        phone_manifest=phone_manifest,
        mfa_manifest=mfa_manifest,
        item_id=item_id,
    )
    write_json(output, report)
    echo_json(report)
