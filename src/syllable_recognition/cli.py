"""Top-level command line interface for the modern pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from syllable_recognition.data.phone_sequences import build_phone_sequence_manifest
from syllable_recognition.data.stage2 import (
    build_aligned_manifest,
    mfa_download_spec,
    mfa_runtime_spec,
    prepare_mfa_inputs,
    record_manual_reviews,
    validate_aligned_manifest,
)


LOGGER = logging.getLogger("sylrec")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@click.group()
def main() -> None:
    """Syllable recognition data, training, evaluation, and inference tools."""
    _configure_logging()


@main.group("train")
def train_group() -> None:
    """Train versioned acoustic tasks after their data gates pass."""


@main.group("align")
def align_group() -> None:
    """Run target-constrained acoustic alignment."""


@main.group("evaluate")
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    click.echo(json.dumps(report, indent=2, sort_keys=True))


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

    result = align_phone_ctc(config_path, checkpoint, audio_path, target_text)
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@train_group.command("phone-ctc")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--overwrite", is_flag=True, help="Explicitly replace a mismatched correctness run.")
def train_phone_ctc_command(config_path: Path, overwrite: bool) -> None:
    """Run the bounded HuBERT Phone CTC correctness experiment."""
    from syllable_recognition.training.phone_ctc import train_phone_ctc

    LOGGER.info(json.dumps({"event": "stage_start", "stage": "phone-ctc-correctness"}))
    report = train_phone_ctc(config_path, overwrite=overwrite)
    LOGGER.info(json.dumps({"event": "stage_complete", "stage": "phone-ctc-correctness", "status": report["status"]}))
    click.echo(json.dumps(report, indent=2, sort_keys=True))


@main.group("data")
def data_group() -> None:
    """Build and validate versioned data artifacts."""


@data_group.command("build-phone-sequences")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--overwrite", is_flag=True, help="Explicitly replace mismatched generated outputs.")
def build_phone_sequences_command(config_path: Path, overwrite: bool) -> None:
    """Build MFA-independent transcript-derived Phone CTC supervision."""
    LOGGER.info(json.dumps({"event": "stage_start", "stage": "phone-sequence-manifest"}))
    report = build_phone_sequence_manifest(config_path, overwrite=overwrite)
    LOGGER.info(json.dumps({"event": "stage_complete", "stage": "phone-sequence-manifest", "status": report["status"]}))
    click.echo(json.dumps(report, indent=2, sort_keys=True))


@data_group.command("prepare-mfa")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--overwrite", is_flag=True, help="Explicitly replace mismatched generated outputs.")
def prepare_mfa_command(config_path: Path, overwrite: bool) -> None:
    """Prepare normalized transcripts, lexicon, and corpus for offline MFA."""
    LOGGER.info(json.dumps({"event": "stage_start", "stage": "prepare-mfa", "config": str(config_path)}))
    report = prepare_mfa_inputs(config_path, overwrite=overwrite)
    LOGGER.info(json.dumps({"event": "stage_complete", "stage": "prepare-mfa", "status": report["status"]}))
    click.echo(json.dumps(report, indent=2, sort_keys=True))


@data_group.command("mfa-spec")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
def mfa_spec_command(config_path: Path) -> None:
    """Print the validated runtime specification consumed by the WSL wrapper."""
    click.echo(json.dumps(mfa_runtime_spec(config_path), indent=2, sort_keys=True))


@data_group.command("mfa-download-spec")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
def mfa_download_spec_command(config_path: Path) -> None:
    """Print the pinned official acoustic-model download contract."""
    click.echo(json.dumps(mfa_download_spec(config_path), indent=2, sort_keys=True))


@data_group.command("build")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--mfa-version", required=True, help="Exact MFA version that produced the TextGrids.")
@click.option("--overwrite", is_flag=True, help="Explicitly replace mismatched generated outputs.")
def build_command(config_path: Path, mfa_version: str, overwrite: bool) -> None:
    """Build a timestamped manifest from existing offline MFA TextGrids."""
    LOGGER.info(json.dumps({"event": "stage_start", "stage": "build-aligned", "config": str(config_path)}))
    report = build_aligned_manifest(config_path, mfa_version=mfa_version, overwrite=overwrite)
    LOGGER.info(json.dumps({"event": "stage_complete", "stage": "build-aligned", "status": report["status"]}))
    click.echo(json.dumps(report, indent=2, sort_keys=True))


@data_group.command("validate")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
def validate_command(config_path: Path) -> None:
    """Validate the aligned manifest and expose the remaining manual gate."""
    report = validate_aligned_manifest(config_path)
    click.echo(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise click.ClickException(f"stage-2 data gate is {report['status']}")


@data_group.command("record-review")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--id", "item_ids", multiple=True, help="Queued audio ID; repeat for multiple items.")
@click.option("--all-items", is_flag=True, help="Apply the decision to every queued item.")
@click.option("--status", type=click.Choice(["pass", "fail"]), required=True)
@click.option("--reviewer", required=True)
@click.option("--method", "review_method", required=True)
@click.option("--evidence", multiple=True, help="Path or identifier for review evidence.")
@click.option("--notes", required=True)
def record_review_command(
    config_path: Path,
    item_ids: tuple[str, ...],
    all_items: bool,
    status: str,
    reviewer: str,
    review_method: str,
    evidence: tuple[str, ...],
    notes: str,
) -> None:
    """Record a pass or failure for manually inspected alignment items."""
    if all_items == bool(item_ids):
        raise click.UsageError("select exactly one of --all-items or one or more --id options")
    report = record_manual_reviews(
        config_path,
        item_ids=None if all_items else item_ids,
        passed=status == "pass",
        reviewer=reviewer,
        review_method=review_method,
        evidence=evidence,
        notes=notes,
    )
    click.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
