"""Evaluate Phone CTC emission spans against MFA without using MFA for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from syllable_recognition.inference.phone_ctc import align_phone_ctc
from syllable_recognition.metrics.alignment import paired_phone_boundary_metrics


def _find_row(path: Path, item_id: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("id") == item_id:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected one row for {item_id}, found {len(matches)}")
    return matches[0]


def evaluate_phone_ctc_against_mfa(
    *,
    training_config: Path,
    checkpoint: Path,
    phone_manifest: Path,
    mfa_manifest: Path,
    item_id: str,
) -> dict[str, Any]:
    phone_row = _find_row(phone_manifest, item_id)
    mfa_row = _find_row(mfa_manifest, item_id)
    root = training_config.resolve().parents[2]
    result = align_phone_ctc(
        training_config,
        checkpoint,
        root / phone_row["audio"],
        phone_row["text"],
    )
    reference = mfa_row["phone_intervals"]
    hypothesis = result["segments"]
    metrics = paired_phone_boundary_metrics(reference, hypothesis)
    return {
        "schema_version": "phone-ctc-vs-mfa-v1",
        "status": "evaluated",
        "item_id": item_id,
        "mode": result["mode"],
        "ctc_timestamp_semantics": result["timestamp_semantics"],
        "mfa_role": "evaluation_reference_only",
        "mfa_timestamps_used_for_training": False,
        "greedy_matches_target": result["greedy_phones"] == result["target_phones"],
        "metrics": metrics,
    }
