"""Download a small, reproducible LibriSpeech sample for stage-0 checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


# 同时固定数据集版本和分片，保证“发布顺序前12条”不会随时间变化。
DATASET_ID = "openslr/librispeech_asr"
DATASET_CONFIG = "all"
DATASET_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
DATASET_SPLIT = "train.clean.100"
DATASET_SHARD = "all/train.clean.100/0000.parquet"
EXPECTED_SAMPLE_RATE = 16_000


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_existing(output_dir: Path, sample_count: int) -> bool:
    report_path = output_dir / "source.json"
    manifest_path = output_dir / "manifest.jsonl"
    if not report_path.is_file() or not manifest_path.is_file():
        return False

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    expected_report = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_shard": DATASET_SHARD,
        "split": DATASET_SPLIT,
        "sample_count": sample_count,
        "selection": "first-n-in-published-split-order",
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        return False
    if len(rows) != sample_count:
        return False

    # 只有来源报告一致还不够，每个落盘FLAC也必须与manifest中的哈希一致。
    for row in rows:
        audio_path = output_dir / row["audio"]
        if not audio_path.is_file() or _sha256(audio_path.read_bytes()) != row["sha256"]:
            return False
    return True


def prepare_sample(output_dir: Path, sample_count: int, overwrite: bool = False) -> dict[str, Any]:
    """Materialize the first N records without changing the official split."""
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    report_path = output_dir / "source.json"
    manifest_path = output_dir / "manifest.jsonl"
    if not overwrite:
        if _validate_existing(output_dir, sample_count):
            return {"status": "skipped", "output": str(output_dir), "sample_count": sample_count}
        if report_path.exists() or manifest_path.exists():
            raise FileExistsError(
                f"Existing output does not match --count {sample_count}; use --overwrite explicitly"
            )

    # 记录旧manifest中的音频，显式覆盖时只清理脚本管理的过期FLAC。
    old_audio_paths: set[Path] = set()
    if overwrite and manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            old_audio_paths.add(output_dir / json.loads(line)["audio"])

    source_parquet = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DATASET_SHARD,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # 源分片很大，只流式读取足够凑齐N条记录的Parquet batch。
    records: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(source_parquet).iter_batches(batch_size=sample_count):
        records.extend(batch.to_pylist())
        if len(records) >= sample_count:
            break

    for record in records[:sample_count]:
        encoded_audio = record["audio"]["bytes"]
        if not encoded_audio:
            raise ValueError(f"Missing encoded audio for {record['id']}")
        # 写入权威本地样本前，先直接校验编码音频的格式。
        info = sf.info(io.BytesIO(encoded_audio))
        if info.samplerate != EXPECTED_SAMPLE_RATE or info.channels != 1:
            raise ValueError(
                f"Unexpected audio format for {record['id']}: "
                f"{info.samplerate} Hz, {info.channels} channels"
            )

        file_name = f"{record['id']}.flac"
        (audio_dir / file_name).write_bytes(encoded_audio)
        rows.append(
            {
                "id": record["id"],
                "audio": f"audio/{file_name}",
                "duration": round(info.frames / info.samplerate, 6),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "text": record["text"],
                "speaker_id": str(record["speaker_id"]),
                "chapter_id": str(record["chapter_id"]),
                "split": DATASET_SPLIT,
                "sha256": _sha256(encoded_audio),
            }
        )

    if len(rows) != sample_count:
        raise RuntimeError(f"Expected {sample_count} samples, received {len(rows)}")

    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    current_audio_paths = {output_dir / row["audio"] for row in rows}
    audio_root = audio_dir.resolve()
    for stale_path in old_audio_paths - current_audio_paths:
        resolved_stale_path = stale_path.resolve()
        # 任何逃出脚本管理audio目录的路径都不得删除。
        if resolved_stale_path.is_relative_to(audio_root) and resolved_stale_path.is_file():
            resolved_stale_path.unlink()
    report = {
        "schema_version": 1,
        "stage": "stage-0-correctness-check",
        "mode": "guided",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_shard": DATASET_SHARD,
        "split": DATASET_SPLIT,
        "sample_count": sample_count,
        "selection": "first-n-in-published-split-order",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": "manifest.jsonl",
    }
    (output_dir / "source.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {"status": "downloaded", "output": str(output_dir), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples/librispeech_train_clean_100"),
    )
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_sample(args.output, args.count, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
