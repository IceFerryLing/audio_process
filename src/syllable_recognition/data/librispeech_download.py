"""Config-driven LibriSpeech split downloader with deterministic subset selection."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


class LibriSpeechDownloadError(RuntimeError):
    """Raised when a configured LibriSpeech download cannot satisfy its contract."""


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    dataset_config: str
    revision: str
    expected_sample_rate: int


@dataclass(frozen=True)
class SplitDownloadSpec:
    name: str
    source_split: str
    source_shards: tuple[str, ...]
    output_dir: Path
    enabled: bool
    target_hours: float | None
    target_items: int | None
    max_seconds_per_speaker: float | None
    minimum_speakers: int

    @property
    def target_seconds(self) -> float | None:
        return None if self.target_hours is None else self.target_hours * 3600.0

    @property
    def selection_mode(self) -> str:
        if self.target_hours is not None:
            return "speaker-capped-published-order-hours-v1"
        if self.target_items is not None:
            return "speaker-capped-published-order-items-v1"
        return "complete-official-split-v1"

    def signature(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_split": self.source_split,
            "source_shards": list(self.source_shards),
            "selection_mode": self.selection_mode,
            "target_hours": self.target_hours,
            "target_items": self.target_items,
            "max_seconds_per_speaker": self.max_seconds_per_speaker,
            "minimum_speakers": self.minimum_speakers,
        }


@dataclass(frozen=True)
class LibriSpeechDownloadConfig:
    source: DatasetSource
    splits: tuple[SplitDownloadSpec, ...]
    report_path: Path
    resolved_config_path: Path
    repository_root: Path
    raw: dict[str, Any]


@dataclass
class SelectionState:
    """Track deterministic subset limits without changing official split membership."""

    spec: SplitDownloadSpec
    selected_items: int = 0
    duration_seconds: float = 0.0
    speaker_seconds: dict[str, float] = field(default_factory=dict)
    chapter_ids: set[str] = field(default_factory=set)

    def can_accept(self, speaker_id: str, duration: float) -> bool:
        if duration <= 0:
            raise LibriSpeechDownloadError("audio duration must be positive")
        cap = self.spec.max_seconds_per_speaker
        if cap is None:
            return True
        return self.speaker_seconds.get(speaker_id, 0.0) + duration <= cap

    def accept(self, speaker_id: str, chapter_id: str, duration: float) -> None:
        if not self.can_accept(speaker_id, duration):
            raise LibriSpeechDownloadError("selection accepted a record beyond the speaker cap")
        self.selected_items += 1
        self.duration_seconds += duration
        self.speaker_seconds[speaker_id] = self.speaker_seconds.get(speaker_id, 0.0) + duration
        self.chapter_ids.add(chapter_id)

    @property
    def complete(self) -> bool:
        if self.spec.target_items is not None:
            return self.selected_items >= self.spec.target_items
        if self.spec.target_seconds is not None:
            return self.duration_seconds >= self.spec.target_seconds
        return False

    def validate_complete(self) -> None:
        if self.selected_items == 0:
            raise LibriSpeechDownloadError(f"{self.spec.name} selected no records")
        if self.spec.target_items is not None and self.selected_items < self.spec.target_items:
            raise LibriSpeechDownloadError(
                f"{self.spec.name} requested {self.spec.target_items} items but selected "
                f"{self.selected_items}"
            )
        if self.spec.target_seconds is not None and self.duration_seconds < self.spec.target_seconds:
            raise LibriSpeechDownloadError(
                f"{self.spec.name} requested {self.spec.target_hours} hours but selected "
                f"{self.duration_seconds / 3600.0:.3f}"
            )
        if len(self.speaker_seconds) < self.spec.minimum_speakers:
            raise LibriSpeechDownloadError(
                f"{self.spec.name} requires {self.spec.minimum_speakers} speakers but selected "
                f"{len(self.speaker_seconds)}"
            )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_output(root: Path, value: str) -> Path:
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise LibriSpeechDownloadError(f"output path escapes the repository: {value}")
    return resolved


def load_librispeech_download_config(path: Path) -> LibriSpeechDownloadConfig:
    """Load and validate the Stage 3 download contract."""
    resolved_config = path.resolve()
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    if raw.get("stage") != "3-librispeech-download" or raw.get("mode") != "guided":
        raise LibriSpeechDownloadError("config must select guided Stage 3 LibriSpeech download")
    root = resolved_config.parents[2]
    dataset = raw["dataset"]
    source = DatasetSource(
        dataset_id=str(dataset["id"]),
        dataset_config=str(dataset["config"]),
        revision=str(dataset["revision"]),
        expected_sample_rate=int(dataset["expected_sample_rate"]),
    )
    splits: list[SplitDownloadSpec] = []
    for name, split in raw["splits"].items():
        target_hours = split.get("target_hours")
        target_items = split.get("target_items")
        if target_hours is not None and target_items is not None:
            raise LibriSpeechDownloadError(f"{name} cannot set both target_hours and target_items")
        if target_hours is not None and float(target_hours) <= 0:
            raise LibriSpeechDownloadError(f"{name}.target_hours must be positive")
        if target_items is not None and int(target_items) < 1:
            raise LibriSpeechDownloadError(f"{name}.target_items must be positive")
        cap = split.get("max_seconds_per_speaker")
        if cap is not None and float(cap) <= 0:
            raise LibriSpeechDownloadError(f"{name}.max_seconds_per_speaker must be positive")
        minimum_speakers = int(split.get("minimum_speakers", 1))
        if minimum_speakers < 1:
            raise LibriSpeechDownloadError(f"{name}.minimum_speakers must be positive")
        splits.append(
            SplitDownloadSpec(
                name=str(name),
                source_split=str(split["source_split"]),
                source_shards=tuple(str(item) for item in split.get("source_shards") or ()),
                output_dir=_resolve_output(root, str(split["output_dir"])),
                enabled=bool(split.get("enabled", True)),
                target_hours=None if target_hours is None else float(target_hours),
                target_items=None if target_items is None else int(target_items),
                max_seconds_per_speaker=None if cap is None else float(cap),
                minimum_speakers=minimum_speakers,
            )
        )
    output = raw["output"]
    return LibriSpeechDownloadConfig(
        source=source,
        splits=tuple(splits),
        report_path=_resolve_output(root, str(output["report"])),
        resolved_config_path=_resolve_output(root, str(output["resolved_config"])),
        repository_root=root,
        raw=raw,
    )


def _existing_matches(
    output_dir: Path,
    source: DatasetSource,
    spec: SplitDownloadSpec,
) -> bool:
    report_path = output_dir / "source.json"
    manifest_path = output_dir / "manifest.jsonl"
    if not report_path.is_file() or not manifest_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_id": source.dataset_id,
        "dataset_config": source.dataset_config,
        "dataset_revision": source.revision,
        "split": spec.name,
        "selection": spec.signature(),
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return False
    if report.get("manifest_sha256") != _sha256_file(manifest_path):
        return False
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != report.get("items"):
        return False
    for row in rows:
        audio_path = output_dir / row["audio"]
        if not audio_path.is_file() or _sha256_file(audio_path) != row["sha256"]:
            return False
    return True


def _discover_split_shards(source: DatasetSource, spec: SplitDownloadSpec) -> list[str]:
    if spec.source_shards:
        expected_prefix = f"{source.dataset_config}/{spec.source_split}/"
        if any(
            not shard.startswith(expected_prefix) or not shard.endswith(".parquet")
            for shard in spec.source_shards
        ):
            raise LibriSpeechDownloadError(
                f"configured shards for {spec.name} must be Parquet files under {expected_prefix}"
            )
        return list(spec.source_shards)

    from huggingface_hub import list_repo_files

    files = list_repo_files(
        repo_id=source.dataset_id,
        repo_type="dataset",
        revision=source.revision,
    )
    prefix = f"{source.dataset_config}/{spec.source_split}/"
    shards = sorted(
        file_name
        for file_name in files
        if file_name.startswith(prefix) and file_name.endswith(".parquet")
    )
    if not shards:
        raise LibriSpeechDownloadError(
            f"no Parquet shards found for {source.dataset_id}:{prefix} at {source.revision}"
        )
    return shards


def _iter_shard_records(source: DatasetSource, shards: Iterable[str]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for shard in shards:
        local_path = hf_hub_download(
            repo_id=source.dataset_id,
            filename=shard,
            repo_type="dataset",
            revision=source.revision,
        )
        for batch in pq.ParquetFile(local_path).iter_batches(batch_size=64):
            yield from batch.to_pylist()


def _managed_audio_paths(manifest_path: Path, output_dir: Path) -> set[Path]:
    if not manifest_path.is_file():
        return set()
    return {
        output_dir / json.loads(line)["audio"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    }


def download_librispeech_split(
    source: DatasetSource,
    spec: SplitDownloadSpec,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize one configured official split or deterministic multi-speaker subset."""
    import soundfile as sf

    report_path = spec.output_dir / "source.json"
    manifest_path = spec.output_dir / "manifest.jsonl"
    if not overwrite and _existing_matches(spec.output_dir, source, spec):
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        return {**existing, "status": "skipped", "output": spec.output_dir.as_posix()}
    if not overwrite and (report_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"existing {spec.name} output does not match the config; use --overwrite explicitly"
        )

    shards = _discover_split_shards(source, spec)
    old_audio_paths = _managed_audio_paths(manifest_path, spec.output_dir) if overwrite else set()
    audio_dir = spec.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    state = SelectionState(spec)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for record in _iter_shard_records(source, shards):
        if state.complete:
            break
        item_id = str(record["id"])
        if item_id in seen_ids:
            raise LibriSpeechDownloadError(f"duplicate LibriSpeech ID across shards: {item_id}")
        seen_ids.add(item_id)
        encoded_audio = record["audio"]["bytes"]
        if not encoded_audio:
            raise LibriSpeechDownloadError(f"{item_id} has no encoded audio")
        info = sf.info(io.BytesIO(encoded_audio))
        if info.samplerate != source.expected_sample_rate or info.channels != 1:
            raise LibriSpeechDownloadError(
                f"{item_id} must be mono {source.expected_sample_rate} Hz; "
                f"found {info.channels} channels at {info.samplerate} Hz"
            )
        duration = info.frames / info.samplerate
        speaker_id = str(record["speaker_id"])
        chapter_id = str(record["chapter_id"])
        if not state.can_accept(speaker_id, duration):
            continue

        file_name = f"{item_id}.flac"
        audio_path = audio_dir / file_name
        audio_path.write_bytes(encoded_audio)
        state.accept(speaker_id, chapter_id, duration)
        rows.append(
            {
                "id": item_id,
                "audio": f"audio/{file_name}",
                "duration": round(duration, 6),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "text": str(record["text"]),
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "split": spec.name,
                "source_split": spec.source_split,
                "sha256": _sha256_bytes(encoded_audio),
            }
        )

    state.validate_complete()
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    current_audio_paths = {spec.output_dir / row["audio"] for row in rows}
    audio_root = audio_dir.resolve()
    for stale_path in old_audio_paths - current_audio_paths:
        resolved_stale = stale_path.resolve()
        if resolved_stale.is_relative_to(audio_root) and resolved_stale.is_file():
            resolved_stale.unlink()

    report = {
        "schema_version": 1,
        "status": "downloaded",
        "stage": "3-librispeech-download",
        "mode": "guided",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": source.dataset_id,
        "dataset_config": source.dataset_config,
        "dataset_revision": source.revision,
        "split": spec.name,
        "source_split": spec.source_split,
        "selection": spec.signature(),
        "items": len(rows),
        "duration_seconds": round(state.duration_seconds, 6),
        "speaker_count": len(state.speaker_seconds),
        "chapter_count": len(state.chapter_ids),
        "speaker_seconds": {
            speaker: round(seconds, 6) for speaker, seconds in sorted(state.speaker_seconds.items())
        },
        "source_shards": shards,
        "manifest": "manifest.jsonl",
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def download_librispeech(
    config_path: Path,
    *,
    split_names: tuple[str, ...] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download all enabled splits or an explicitly requested configured split list."""
    config = load_librispeech_download_config(config_path)
    available = {spec.name: spec for spec in config.splits}
    unknown = set(split_names) - set(available)
    if unknown:
        raise LibriSpeechDownloadError(f"unknown configured splits: {sorted(unknown)}")
    selected = (
        [available[name] for name in split_names]
        if split_names
        else [spec for spec in config.splits if spec.enabled]
    )
    if not selected:
        raise LibriSpeechDownloadError("no LibriSpeech splits are enabled or selected")

    reports = {
        spec.name: download_librispeech_split(config.source, spec, overwrite=overwrite)
        for spec in selected
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    config.resolved_config_path.write_text(
        yaml.safe_dump(config.raw, sort_keys=True),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "stage": "3-librispeech-download",
        "mode": "guided",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _sha256_file(config_path),
        "splits": reports,
    }
    config.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DatasetSource",
    "LibriSpeechDownloadConfig",
    "LibriSpeechDownloadError",
    "SelectionState",
    "SplitDownloadSpec",
    "download_librispeech",
    "download_librispeech_split",
    "load_librispeech_download_config",
]
