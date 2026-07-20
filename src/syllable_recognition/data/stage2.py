"""Stage-2 preparation of MFA inputs and untimed syllable supervision."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from syllable_recognition.core.artifacts import (
    read_json,
    read_jsonl as _read_jsonl,
    sha256_bytes as _sha256_bytes,
    sha256_file,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)

from .normalization import NORMALIZER_VERSION, normalize_librispeech_text
from .alignment import AlignmentParseError, align_prepared_row
from .pronunciation import Pronunciation, PronunciationResolver
from .syllabification import Syllabifier


class Stage2DataError(RuntimeError):
    """Raised when a stage-2 input or output gate fails."""


PENDING_VISUAL_REVIEW = "pending_visual_review"
PASSED_VISUAL_REVIEW = "passed_visual_review"
FAILED_VISUAL_REVIEW = "failed_visual_review"
MANUAL_REVIEW_STATUSES = {
    PENDING_VISUAL_REVIEW,
    PASSED_VISUAL_REVIEW,
    FAILED_VISUAL_REVIEW,
}


@dataclass(frozen=True)
class Stage2Config:
    path: Path
    root: Path
    raw: Mapping[str, Any]

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()


def load_stage2_config(path: Path) -> Stage2Config:
    resolved_path = path.resolve()
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if raw.get("stage") != "2-data-labeling" or raw.get("mode") != "guided":
        raise Stage2DataError("stage-2 config must explicitly select guided data labeling")
    root = resolved_path.parents[2]
    expected_split = raw["input"]["split"]
    if expected_split != "train.clean.100":
        raise Stage2DataError("the stage-0 sample must retain the official train.clean.100 split")
    if raw["normalization"]["version"] != NORMALIZER_VERSION:
        raise Stage2DataError("normalizer version differs from the implementation")
    return Stage2Config(resolved_path, root, raw)


def _validate_input_row(row: Mapping[str, Any], expected_split: str, audio_path: Path) -> None:
    required = {"id", "audio", "duration", "text", "speaker_id", "sample_rate", "channels", "split", "sha256"}
    missing = required - set(row)
    if missing:
        raise Stage2DataError(f"input row is missing fields: {sorted(missing)}")
    if row["split"] != expected_split:
        raise Stage2DataError(f"{row['id']} changed official split")
    if row["sample_rate"] != 16000 or row["channels"] != 1:
        raise Stage2DataError(f"{row['id']} is not mono 16 kHz")
    if not audio_path.is_file() or sha256_file(audio_path) != row["sha256"]:
        raise Stage2DataError(f"{row['id']} audio is missing or has a hash mismatch")


def _serialize_pronunciation(pronunciation: Pronunciation, syllabifier: Syllabifier) -> dict[str, Any]:
    syllables = syllabifier.syllabify(pronunciation.phones)
    return {
        "word": pronunciation.word,
        "phones": list(pronunciation.phones),
        "source": pronunciation.source,
        "variant_index": pronunciation.variant_index,
        "variant_count": pronunciation.variant_count,
        "syllables": [syllable.to_dict() for syllable in syllables],
    }


def _prepared_outputs_match(config: Stage2Config, input_hash: str, config_hash: str) -> bool:
    report_path = config.resolve(config.raw["output"]["prepare_report"])
    if not report_path.is_file():
        return False
    report = read_json(report_path)
    required_paths = [
        config.resolve(config.raw["output"]["prepared_manifest"]),
        config.resolve(config.raw["output"]["oov_report"]),
        config.resolve(config.raw["mfa"]["dictionary_path"]),
        config.resolve(config.raw["mfa"]["corpus_directory"]),
    ]
    return (
        report.get("input_manifest_sha256") == input_hash
        and report.get("config_sha256") == config_hash
        and all(path.exists() for path in required_paths)
    )


def prepare_mfa_inputs(config_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Build an MFA corpus, selected-pronunciation lexicon, and untimed manifest."""
    config = load_stage2_config(config_path)
    input_manifest = config.resolve(config.raw["input"]["manifest"])
    input_hash = sha256_file(input_manifest)
    config_hash = _sha256_bytes(config.path.read_bytes())
    if not overwrite and _prepared_outputs_match(config, input_hash, config_hash):
        return {"status": "skipped", "stage": "prepare-mfa", "input_items": len(_read_jsonl(input_manifest))}

    output_paths = [
        config.resolve(config.raw["output"]["prepared_manifest"]),
        config.resolve(config.raw["output"]["prepare_report"]),
        config.resolve(config.raw["mfa"]["dictionary_path"]),
    ]
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError("existing stage-2 preparation does not match; use --overwrite explicitly")

    rows = _read_jsonl(input_manifest)
    expected_items = config.raw["data_gate"]["expected_items"]
    if len(rows) != expected_items:
        raise Stage2DataError(f"expected {expected_items} input rows, received {len(rows)}")

    resolver = PronunciationResolver.from_default_packages()
    syllabifier = Syllabifier.from_config(config.resolve(config.raw["syllabification"]["config"]))
    corpus_dir = config.resolve(config.raw["mfa"]["corpus_directory"])
    corpus_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows: list[dict[str, Any]] = []
    selected: dict[str, Pronunciation] = {}

    for row in rows:
        source_audio = input_manifest.parent / row["audio"]
        _validate_input_row(row, config.raw["input"]["split"], source_audio)
        normalized = normalize_librispeech_text(row["text"])
        pronunciations = resolver.resolve_many(normalized.words)
        for pronunciation in pronunciations:
            selected.setdefault(pronunciation.word, pronunciation)

        speaker_dir = corpus_dir / str(row["speaker_id"])
        speaker_dir.mkdir(parents=True, exist_ok=True)
        target_audio = speaker_dir / f"{row['id']}{source_audio.suffix.lower()}"
        target_label = speaker_dir / f"{row['id']}.lab"
        if overwrite or not target_audio.exists():
            shutil.copy2(source_audio, target_audio)
        target_label.write_text(normalized.normalized + "\n", encoding="utf-8")

        words = [_serialize_pronunciation(pronunciation, syllabifier) for pronunciation in pronunciations]
        prepared_rows.append(
            {
                **row,
                "audio": source_audio.relative_to(config.root).as_posix(),
                "mode": "guided",
                "normalized_text": normalized.normalized,
                "normalizer_version": normalized.version,
                "syllabifier_version": syllabifier.rule_version,
                "words": words,
                "phones": [phone for word in words for phone in word["phones"]],
                "syllables": [syllable for word in words for syllable in word["syllables"]],
                "alignment_status": "pending_mfa",
            }
        )

    dictionary_path = config.resolve(config.raw["mfa"]["dictionary_path"])
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_path.write_text(
        "".join(f"{word}\t{' '.join(selected[word].phones)}\n" for word in sorted(selected)),
        encoding="utf-8",
    )
    prepared_manifest = config.resolve(config.raw["output"]["prepared_manifest"])
    _write_jsonl(prepared_manifest, prepared_rows)

    oov_rows = [
        {
            "word": word,
            "phones": list(pronunciation.phones),
            "source": pronunciation.source,
        }
        for word, pronunciation in sorted(selected.items())
        if pronunciation.source == resolver.g2p_name
    ]
    _write_jsonl(config.resolve(config.raw["output"]["oov_report"]), oov_rows)
    resolved_config = config.resolve(config.raw["output"]["resolved_config"])
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    resolved_config.write_text(yaml.safe_dump(dict(config.raw), sort_keys=True), encoding="utf-8")

    report = {
        "schema_version": 1,
        "stage": "2-data-labeling-prepare-mfa",
        "mode": "guided",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": config.raw["input"]["manifest"],
        "input_manifest_sha256": input_hash,
        "config_sha256": config_hash,
        "split": config.raw["input"]["split"],
        "input_items": len(rows),
        "unique_words": len(selected),
        "oov_words": len(oov_rows),
        "oov_rate": len(oov_rows) / len(selected),
        "dictionary": resolver.dictionary_name,
        "g2p": resolver.g2p_name,
        "syllabifier": syllabifier.rule_version,
        "prepared_manifest_sha256": sha256_file(prepared_manifest),
        "dictionary_sha256": sha256_file(dictionary_path),
    }
    _write_json(config.resolve(config.raw["output"]["prepare_report"]), report)
    return report


def _textgrid_set_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _find_textgrid(directory: Path, audio_id: str) -> Path:
    matches = list(directory.rglob(f"{audio_id}.TextGrid"))
    if len(matches) != 1:
        raise AlignmentParseError(f"expected one TextGrid for {audio_id}, found {len(matches)}")
    return matches[0]


def mfa_runtime_spec(config_path: Path) -> dict[str, Any]:
    """Resolve and validate the external MFA paths without running MFA."""
    config = load_stage2_config(config_path)
    mfa_config = config.raw["mfa"]
    archive = config.resolve(mfa_config["acoustic_model_archive"])
    if not archive.is_file():
        raise Stage2DataError(f"MFA acoustic model archive is missing: {archive}")
    if archive.stat().st_size != mfa_config["acoustic_model_size_bytes"]:
        raise Stage2DataError("MFA acoustic model archive size does not match the release metadata")
    if sha256_file(archive) != mfa_config["acoustic_model_sha256"]:
        raise Stage2DataError("MFA acoustic model archive hash does not match the pinned config")
    prefix = mfa_config["environment_prefix"]
    if not isinstance(prefix, str) or not prefix.startswith("~/"):
        raise Stage2DataError("MFA environment_prefix must be a portable home-relative path")
    micromamba = mfa_config["micromamba_executable"]
    if not isinstance(micromamba, str) or not micromamba.startswith("~/"):
        raise Stage2DataError("micromamba_executable must be a portable home-relative path")
    wsl_prefix = f"$HOME/{prefix[2:]}"
    wsl_micromamba = f"$HOME/{micromamba[2:]}"
    return {
        "config": str(config.path),
        "corpus_directory": str(config.resolve(mfa_config["corpus_directory"])),
        "dictionary_path": str(config.resolve(mfa_config["dictionary_path"])),
        "acoustic_model_archive": str(archive),
        "textgrid_directory": str(config.resolve(mfa_config["textgrid_directory"])),
        "mfa_runner": f"{wsl_micromamba} run -p {wsl_prefix} mfa",
        "mfa_version": str(mfa_config["version"]),
        "output_format": mfa_config["output_format"],
        "expected_textgrids": config.raw["data_gate"]["expected_items"],
    }


def mfa_download_spec(config_path: Path) -> dict[str, Any]:
    """Return the pinned official acoustic-model download contract."""
    config = load_stage2_config(config_path)
    mfa_config = config.raw["mfa"]
    return {
        "url": mfa_config["acoustic_model_url"],
        "destination": str(config.resolve(mfa_config["acoustic_model_archive"])),
        "size_bytes": mfa_config["acoustic_model_size_bytes"],
        "sha256": mfa_config["acoustic_model_sha256"],
        "release": mfa_config["acoustic_model_release"],
    }


def build_aligned_manifest(
    config_path: Path,
    *,
    mfa_version: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert MFA TextGrids to timestamped syllable supervision and gate reports."""
    config = load_stage2_config(config_path)
    if str(config.raw["mfa"]["version"]) != mfa_version:
        raise Stage2DataError(
            f"MFA version {mfa_version!r} differs from configured {config.raw['mfa']['version']!r}"
        )
    prepared_path = config.resolve(config.raw["output"]["prepared_manifest"])
    prepare_report_path = config.resolve(config.raw["output"]["prepare_report"])
    if not prepared_path.is_file() or not prepare_report_path.is_file():
        raise Stage2DataError("prepare-mfa outputs are missing")

    config_hash = _sha256_bytes(config.path.read_bytes())
    prepared_hash = sha256_file(prepared_path)
    prepare_report = read_json(prepare_report_path)
    if prepare_report.get("config_sha256") != config_hash:
        raise Stage2DataError("prepare-mfa outputs were built with a different config")
    if prepare_report.get("prepared_manifest_sha256") != prepared_hash:
        raise Stage2DataError("prepared manifest hash differs from its report")

    aligned_path = config.resolve(config.raw["output"]["aligned_manifest"])
    build_report_path = config.resolve(config.raw["output"]["build_report"])
    textgrid_dir = config.resolve(config.raw["mfa"]["textgrid_directory"])
    prepared_rows = _read_jsonl(prepared_path)
    textgrid_paths = [_find_textgrid(textgrid_dir, row["id"]) for row in prepared_rows]
    textgrid_hash = _textgrid_set_hash(textgrid_paths)

    if not overwrite and build_report_path.is_file() and aligned_path.is_file():
        old_report = read_json(build_report_path)
        if (
            old_report.get("config_sha256") == config_hash
            and old_report.get("prepared_manifest_sha256") == prepared_hash
            and old_report.get("textgrid_set_sha256") == textgrid_hash
            and old_report.get("aligned_manifest_sha256") == sha256_file(aligned_path)
        ):
            return {"status": "skipped", "stage": "build-aligned", "input_items": len(prepared_rows)}
        raise FileExistsError("existing aligned outputs do not match; use --overwrite explicitly")

    aligned_rows: list[dict[str, Any]] = []
    item_reports: list[dict[str, Any]] = []
    mfa_config = config.raw["mfa"]
    for row, textgrid_path in zip(prepared_rows, textgrid_paths):
        try:
            aligned = align_prepared_row(
                row,
                textgrid_path,
                word_tier=mfa_config["word_tier"],
                phone_tier=mfa_config["phone_tier"],
                silence_phones=set(mfa_config["silence_phones"]),
            )
            aligned["mfa_textgrid"] = textgrid_path.relative_to(config.root).as_posix()
            aligned["mfa_version"] = mfa_version
            aligned["mfa_acoustic_model"] = mfa_config["acoustic_model"]
            aligned_rows.append(aligned)
            item_reports.append(
                {
                    "id": row["id"],
                    "status": "aligned",
                    "word_count": len(aligned["words"]),
                    "phone_count": len(aligned["phone_intervals"]),
                    "syllable_count": len(aligned["syllables"]),
                    "textgrid": aligned["mfa_textgrid"],
                }
            )
        except (AlignmentParseError, ValueError) as error:
            item_reports.append({"id": row["id"], "status": "failed", "error": str(error)})

    _write_jsonl(aligned_path, aligned_rows)
    _write_jsonl(config.resolve(config.raw["output"]["item_report"]), item_reports)
    review_count = min(config.raw["data_gate"]["minimum_manual_review_items"], len(aligned_rows))
    review_rows = [
        {
            "id": row["id"],
            "audio": row["audio"],
            "textgrid": row["mfa_textgrid"],
            "normalized_text": row["normalized_text"],
            "word_count": len(row["words"]),
            "phone_count": len(row["phone_intervals"]),
            "syllable_count": len(row["syllables"]),
            "automated_checks": {
                "word_sequence_exact": True,
                "phone_sequence_exact": True,
                "timestamps_monotonic": True,
                "timestamps_within_audio": True,
                "every_syllable_has_nucleus": True,
            },
            "review_status": PENDING_VISUAL_REVIEW,
        }
        for row in aligned_rows[:review_count]
    ]
    _write_jsonl(config.resolve(config.raw["output"]["manual_review"]), review_rows)

    success_rate = len(aligned_rows) / len(prepared_rows) if prepared_rows else 0.0
    automated_passed = success_rate >= config.raw["data_gate"]["required_alignment_success_rate"]
    report = {
        "schema_version": 1,
        "stage": "2-data-labeling-build",
        "mode": "guided",
        "status": "passed_automated" if automated_passed else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": config.raw["input"]["split"],
        "config_sha256": config_hash,
        "prepared_manifest_sha256": prepared_hash,
        "textgrid_set_sha256": textgrid_hash,
        "aligned_manifest_sha256": sha256_file(aligned_path),
        "input_items": len(prepared_rows),
        "aligned_items": len(aligned_rows),
        "failed_items": len(prepared_rows) - len(aligned_rows),
        "alignment_success_rate": success_rate,
        "total_words": sum(len(row["words"]) for row in aligned_rows),
        "total_phones": sum(len(row["phone_intervals"]) for row in aligned_rows),
        "total_syllables": sum(len(row["syllables"]) for row in aligned_rows),
        "manual_review_items": len(review_rows),
        "manual_review_status": "pending",
        "mfa_version": mfa_version,
        "mfa_acoustic_model": mfa_config["acoustic_model"],
        "downstream_training_allowed": False,
    }
    _write_json(build_report_path, report)
    if not automated_passed and config.raw["data_gate"]["stop_on_failure"]:
        raise Stage2DataError(
            f"alignment success rate {success_rate:.3f} is below the configured gate"
        )
    return report


def record_manual_reviews(
    config_path: Path,
    *,
    item_ids: tuple[str, ...] | None,
    passed: bool,
    reviewer: str,
    review_method: str,
    evidence: tuple[str, ...],
    notes: str,
) -> dict[str, Any]:
    """Record an explicit, auditable human decision for queued review items."""
    config = load_stage2_config(config_path)
    review_path = config.resolve(config.raw["output"]["manual_review"])
    if not review_path.is_file():
        raise Stage2DataError("manual review queue is missing; run data build first")
    if not reviewer.strip() or not review_method.strip() or not notes.strip():
        raise Stage2DataError("reviewer, review method, and notes must be non-empty")

    reviews = _read_jsonl(review_path)
    queued_ids = [str(row["id"]) for row in reviews]
    selected_ids = set(queued_ids if item_ids is None else item_ids)
    if not selected_ids:
        raise Stage2DataError("at least one queued item must be selected")
    unknown = selected_ids - set(queued_ids)
    if unknown:
        raise Stage2DataError(f"manual review IDs are not queued: {sorted(unknown)}")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    status = PASSED_VISUAL_REVIEW if passed else FAILED_VISUAL_REVIEW
    updated: list[dict[str, Any]] = []
    for row in reviews:
        if row["id"] in selected_ids:
            row = {
                **row,
                "review_status": status,
                "reviewer": reviewer.strip(),
                "review_method": review_method.strip(),
                "reviewed_at": reviewed_at,
                "evidence": list(evidence),
                "review_notes": notes.strip(),
            }
        updated.append(row)
    _write_jsonl(review_path, updated)
    return {
        "status": "recorded",
        "review_status": status,
        "updated_items": len(selected_ids),
        "review_queue": config.raw["output"]["manual_review"],
    }


def summarize_manual_reviews(
    reviews: list[dict[str, Any]],
    *,
    aligned_ids: set[str],
    minimum_review: int,
) -> dict[str, Any]:
    """Apply the strict manual gate: only explicit passes count as passing."""
    review_ids = [str(row.get("id", "")) for row in reviews]
    review_problems: list[str] = []
    if len(review_ids) != len(set(review_ids)):
        review_problems.append("manual review queue contains duplicate IDs")
    unknown_ids = set(review_ids) - aligned_ids
    if unknown_ids:
        review_problems.append(f"manual review queue contains unknown IDs: {sorted(unknown_ids)}")

    invalid_status_ids = [
        str(row.get("id", ""))
        for row in reviews
        if row.get("review_status") not in MANUAL_REVIEW_STATUSES
    ]
    if invalid_status_ids:
        review_problems.append(f"manual review queue contains invalid statuses: {invalid_status_ids}")

    passed_ids = [str(row["id"]) for row in reviews if row.get("review_status") == PASSED_VISUAL_REVIEW]
    failed_ids = [str(row["id"]) for row in reviews if row.get("review_status") == FAILED_VISUAL_REVIEW]
    pending_ids = [str(row["id"]) for row in reviews if row.get("review_status") == PENDING_VISUAL_REVIEW]
    gate_passed = (
        len(passed_ids) >= minimum_review
        and not failed_ids
        and not pending_ids
        and not review_problems
    )
    return {
        "passed_ids": passed_ids,
        "failed_ids": failed_ids,
        "pending_ids": pending_ids,
        "review_problems": review_problems,
        "manual_gate_passed": gate_passed,
    }


def validate_aligned_manifest(config_path: Path) -> dict[str, Any]:
    """Re-run manifest invariants and report the remaining manual gate."""
    config = load_stage2_config(config_path)
    aligned_path = config.resolve(config.raw["output"]["aligned_manifest"])
    build_report_path = config.resolve(config.raw["output"]["build_report"])
    review_path = config.resolve(config.raw["output"]["manual_review"])
    if not aligned_path.is_file() or not build_report_path.is_file() or not review_path.is_file():
        raise Stage2DataError("aligned manifest, build report, or manual review queue is missing")
    build_report = read_json(build_report_path)
    if build_report["aligned_manifest_sha256"] != sha256_file(aligned_path):
        raise Stage2DataError("aligned manifest hash differs from the build report")

    rows = _read_jsonl(aligned_path)
    problems: list[str] = []
    for row in rows:
        if row["split"] != config.raw["input"]["split"] or row["alignment_status"] != "aligned":
            problems.append(f"{row['id']}: split or alignment status mismatch")
        if [item["phone"] for item in row["phone_intervals"]] != row["phones"]:
            problems.append(f"{row['id']}: phone interval sequence mismatch")
        reconstructed = [phone for syllable in row["syllables"] for phone in syllable["phones"]]
        if reconstructed != row["phones"]:
            problems.append(f"{row['id']}: syllable sequence is not lossless")
        previous_end = 0.0
        for syllable in row["syllables"]:
            if not (0 <= syllable["start"] < syllable["end"] <= row["duration"]):
                problems.append(f"{row['id']}: syllable timestamp is outside the audio")
            if syllable["start"] < previous_end - 1e-5:
                problems.append(f"{row['id']}: syllable timestamps overlap")
            previous_end = syllable["end"]

    reviews = _read_jsonl(review_path)
    minimum_review = config.raw["data_gate"]["minimum_manual_review_items"]
    manual = summarize_manual_reviews(
        reviews,
        aligned_ids={str(row["id"]) for row in rows},
        minimum_review=minimum_review,
    )
    problems.extend(manual["review_problems"])
    manually_reviewed = len(manual["passed_ids"]) + len(manual["failed_ids"])
    if manual["failed_ids"]:
        problems.append(f"manual visual review failed for IDs: {manual['failed_ids']}")
    if problems:
        status = "failed"
    elif manual["manual_gate_passed"]:
        status = "passed"
    else:
        status = "pending_manual_review"
    return {
        "status": status,
        "items": len(rows),
        "problems": problems,
        "manual_review_items": len(reviews),
        "manually_reviewed_items": manually_reviewed,
        "passed_manual_review_items": len(manual["passed_ids"]),
        "failed_manual_review_items": len(manual["failed_ids"]),
        "pending_manual_review_items": len(manual["pending_ids"]),
        "minimum_manual_review_items": minimum_review,
        "manual_gate_passed": manual["manual_gate_passed"],
        "downstream_training_allowed": not problems and manual["manual_gate_passed"],
    }
