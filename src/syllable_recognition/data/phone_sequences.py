"""Build MFA-independent phone-sequence supervision for Phone CTC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml

from syllable_recognition.core.artifacts import (
    read_json,
    read_jsonl as _read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)

from .normalization import NORMALIZER_VERSION, normalize_librispeech_text
from .pronunciation import PronunciationResolver


class PhoneSequenceDataError(RuntimeError):
    """Raised when sequence-only Phone CTC supervision fails its data gate."""


@dataclass(frozen=True)
class PhoneSequenceConfig:
    path: Path
    root: Path
    raw: Mapping[str, Any]

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()


@dataclass(frozen=True)
class PhoneVocabulary:
    tokens: tuple[str, ...]
    blank_id: int = 0
    unknown_id: int = 1
    label_padding_id: int = -100

    @classmethod
    def from_training_sequences(cls, sequences: Sequence[Sequence[str]]) -> "PhoneVocabulary":
        phones = sorted({phone for sequence in sequences for phone in sequence})
        return cls(("<blank>", "<unk>", *phones))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, phones: Sequence[str]) -> list[int]:
        mapping = self.token_to_id
        return [mapping.get(phone, self.unknown_id) for phone in phones]

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.tokens[index] if 0 <= index < len(self.tokens) else "<unk>" for index in ids]

    def to_dict(self, *, source_split: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": "phone_ctc",
            "source_split": source_split,
            "vocabulary_source": "train_only",
            "stress_preserved": True,
            "tokens": list(self.tokens),
            "token_to_id": self.token_to_id,
            "blank_id": self.blank_id,
            "unknown_id": self.unknown_id,
            "label_padding_id": self.label_padding_id,
        }


def minimum_ctc_frames(labels: Sequence[str] | Sequence[int]) -> int:
    """Return target length plus one blank frame for each adjacent repeat."""
    repeats = sum(left == right for left, right in zip(labels, labels[1:]))
    return len(labels) + repeats


def convolution_output_length(
    input_samples: int,
    kernels: Sequence[int],
    strides: Sequence[int],
) -> int:
    if input_samples < 1 or len(kernels) != len(strides):
        raise ValueError("invalid convolution length specification")
    length = input_samples
    for kernel, stride in zip(kernels, strides):
        length = (length - kernel) // stride + 1
        if length < 1:
            return 0
    return length


def load_phone_sequence_config(path: Path) -> PhoneSequenceConfig:
    resolved = path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if raw.get("stage") != "3-phone-sequence-manifest" or raw.get("task") != "phone_ctc":
        raise PhoneSequenceDataError("config must select the phone-sequence manifest stage")
    if raw.get("mode") not in {"guided", "open"}:
        raise PhoneSequenceDataError("config must explicitly select guided or open mode")
    if raw["labels"]["normalization_version"] != NORMALIZER_VERSION:
        raise PhoneSequenceDataError("normalization version differs from the implementation")
    if raw["supervision"].get("mfa_timestamps_used") is not False:
        raise PhoneSequenceDataError("Phone CTC sequence supervision must not use MFA timestamps")
    return PhoneSequenceConfig(resolved, resolved.parents[2], raw)


def _existing_outputs_match(
    config: PhoneSequenceConfig,
    *,
    input_hash: str,
    config_hash: str,
) -> bool:
    report_path = config.resolve(config.raw["output"]["report"])
    manifest_path = config.resolve(config.raw["output"]["manifest"])
    vocabulary_path = config.resolve(config.raw["output"]["vocabulary"])
    if not report_path.is_file() or not manifest_path.is_file() or not vocabulary_path.is_file():
        return False
    report = read_json(report_path)
    return (
        report.get("input_manifest_sha256") == input_hash
        and report.get("config_sha256") == config_hash
        and report.get("output_manifest_sha256") == sha256_file(manifest_path)
        and report.get("vocabulary_sha256") == sha256_file(vocabulary_path)
    )


def build_phone_sequence_manifest(config_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Generate ordered phone labels directly from transcripts, without MFA boundaries."""
    config = load_phone_sequence_config(config_path)
    input_path = config.resolve(config.raw["input"]["manifest"])
    output_path = config.resolve(config.raw["output"]["manifest"])
    vocabulary_path = config.resolve(config.raw["output"]["vocabulary"])
    report_path = config.resolve(config.raw["output"]["report"])
    input_hash = sha256_file(input_path)
    config_hash = sha256_bytes(config.path.read_bytes())
    if not overwrite and _existing_outputs_match(config, input_hash=input_hash, config_hash=config_hash):
        return {"status": "skipped", "stage": "phone-sequence-manifest"}
    if not overwrite and any(path.exists() for path in (output_path, vocabulary_path, report_path)):
        raise FileExistsError("existing phone-sequence outputs do not match; use --overwrite explicitly")

    source_rows = _read_jsonl(input_path)
    if len(source_rows) != config.raw["data_gate"]["expected_items"]:
        raise PhoneSequenceDataError("input item count differs from the configured correctness gate")

    model_config_path = config.resolve(config.raw["encoder"]["local_path"]) / "config.json"
    model_config = read_json(model_config_path)
    kernels = model_config["conv_kernel"]
    strides = model_config["conv_stride"]
    resolver = PronunciationResolver.from_default_packages()
    rows: list[dict[str, Any]] = []
    all_sequences: list[list[str]] = []
    oov_words: set[str] = set()

    for source in source_rows:
        if source["split"] != config.raw["input"]["split"]:
            raise PhoneSequenceDataError(f"{source['id']} changed official split")
        source_audio = input_path.parent / source["audio"]
        if not source_audio.is_file() or sha256_file(source_audio) != source["sha256"]:
            raise PhoneSequenceDataError(f"{source['id']} audio is missing or has a hash mismatch")
        info = sf.info(source_audio)
        if info.samplerate != 16_000 or info.channels != 1:
            raise PhoneSequenceDataError(f"{source['id']} is not mono 16 kHz")

        normalized = normalize_librispeech_text(source["text"])
        pronunciations = resolver.resolve_many(normalized.words)
        phones = [phone for pronunciation in pronunciations for phone in pronunciation.phones]
        for pronunciation in pronunciations:
            if pronunciation.source.startswith("g2p_en-"):
                oov_words.add(pronunciation.word)
        output_frames = convolution_output_length(info.frames, kernels, strides)
        minimum_frames = minimum_ctc_frames(phones)
        if output_frames < minimum_frames:
            raise PhoneSequenceDataError(
                f"{source['id']} is CTC-infeasible: {output_frames} frames < {minimum_frames} required"
            )
        all_sequences.append(phones)
        rows.append(
            {
                "id": source["id"],
                "audio": source_audio.relative_to(config.root).as_posix(),
                "duration": source["duration"],
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "speaker_id": source["speaker_id"],
                "chapter_id": source["chapter_id"],
                "split": source["split"],
                "mode": config.raw["mode"],
                "task": "phone_ctc",
                "text": source["text"],
                "normalized_text": normalized.normalized,
                "phones": phones,
                "phone_count": len(phones),
                "encoder_output_frames": output_frames,
                "minimum_ctc_frames": minimum_frames,
                "audio_sha256": source["sha256"],
                "label_source": "transcript+cmudict/g2p",
                "mfa_timestamps_used": False,
            }
        )

    vocabulary = PhoneVocabulary.from_training_sequences(all_sequences)
    _write_jsonl(output_path, rows)
    _write_json(vocabulary_path, vocabulary.to_dict(source_split=config.raw["input"]["split"]))
    resolved_config_path = config.resolve(config.raw["output"]["resolved_config"])
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path.write_text(yaml.safe_dump(dict(config.raw), sort_keys=True), encoding="utf-8")

    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "3-phone-sequence-manifest",
        "task": "phone_ctc",
        "mode": config.raw["mode"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": config.raw["input"]["split"],
        "items": len(rows),
        "total_duration_seconds": sum(float(row["duration"]) for row in rows),
        "total_phones": sum(int(row["phone_count"]) for row in rows),
        "vocabulary_size": len(vocabulary.tokens),
        "oov_words": sorted(oov_words),
        "ctc_infeasible_items": 0,
        "mfa_timestamps_used": False,
        "mfa_role": "evaluation_only",
        "input_manifest_sha256": input_hash,
        "config_sha256": config_hash,
        "output_manifest_sha256": sha256_file(output_path),
        "vocabulary_sha256": sha256_file(vocabulary_path),
    }
    _write_json(report_path, report)
    return report
