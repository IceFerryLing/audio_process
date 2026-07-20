"""Download and verify the pinned HuBERT Base encoder used by this project."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


# 将Hub版本、文件集合、字节数和哈希视为一个不可拆分的模型资产契约。
MODEL_ID = "facebook/hubert-base-ls960"
MODEL_REVISION = "dba3bb02fda4248b6e082697eee756de8fe8aa8a"
EXPECTED_FILES = {
    "config.json": {
        "sha256": "56be398848bbd9cbc720172b0d45b2f01cac7652f2968f0f31f5faf9f2986acc",
        "size_bytes": 1385,
    },
    "preprocessor_config.json": {
        "sha256": "4a93853b74278b7c769d07f5a861e5d12ceb5db2bced5620d335f87238cb9e86",
        "size_bytes": 213,
    },
    "pytorch_model.bin": {
        "sha256": "062249fffb353eab67547a2fbc129f7c31a2f459faf641b19e8fb007cc5c48ad",
        "size_bytes": 377_569_754,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    # 流式计算377 MB权重的哈希，避免一次性读入内存。
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(output: Path) -> tuple[bool, dict[str, Any]]:
    actual: dict[str, Any] = {}
    valid = True
    for name, expected in EXPECTED_FILES.items():
        path = output / name
        if not path.is_file():
            valid = False
            actual[name] = {"status": "missing"}
            continue
        details = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        details["status"] = "valid" if details == expected else "mismatch"
        valid = valid and details["status"] == "valid"
        actual[name] = details
    return valid, actual


def download_model(output: Path, overwrite: bool = False) -> dict[str, Any]:
    valid, files = validate_model(output)
    # 只有全部固定文件都存在且完全匹配时，幂等重跑才允许跳过。
    if valid and not overwrite:
        return {"status": "skipped", "model_id": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    if output.exists() and not overwrite:
        raise FileExistsError("existing HuBERT files are incomplete or mismatched; use --overwrite explicitly")

    output.mkdir(parents=True, exist_ok=True)
    # 只下载契约内文件，避免额外拉取tokenizer或其他框架权重。
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=output,
        allow_patterns=list(EXPECTED_FILES),
    )
    # 网络请求成功不代表资产合格，落盘后必须再次校验实际字节。
    valid, files = validate_model(output)
    if not valid:
        raise RuntimeError(f"downloaded HuBERT files failed verification: {files}")

    report = {
        "schema_version": 1,
        "stage": "stage-0-model-asset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "framework": "pytorch",
        "sample_rate": 16_000,
        "files": files,
    }
    (output / "source.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {"status": "downloaded", **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/models/facebook-hubert-base-ls960"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download_model(args.output, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
