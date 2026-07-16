"""Render waveform and MFA word-boundary sheets for Stage 2 visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_review_sheets(manifest: Path, output_dir: Path, count: int = 10) -> list[Path]:
    rows = _read_jsonl(manifest)[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for sheet_index, start in enumerate(range(0, len(rows), 5), start=1):
        sheet_rows = rows[start : start + 5]
        figure, axes = plt.subplots(len(sheet_rows), 1, figsize=(20, 12), constrained_layout=True)
        if len(sheet_rows) == 1:
            axes = [axes]
        for axis, row in zip(axes, sheet_rows):
            audio, sample_rate = sf.read(str(row["audio"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            stride = max(1, sample_rate // 1000)
            samples = audio[::stride]
            times = np.arange(len(samples)) * stride / sample_rate
            axis.plot(times, samples, color="#1f4e79", linewidth=0.45)
            axis.axhline(0, color="#777777", linewidth=0.3)
            for word_index, word in enumerate(row["words"]):
                start_time = float(word["start"])
                end_time = float(word["end"])
                axis.axvline(start_time, color="#b23a48", linewidth=0.45, alpha=0.8)
                axis.text(
                    (start_time + end_time) / 2,
                    1.03 if word_index % 2 == 0 else 0.88,
                    str(word["word"]),
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=55,
                )
            axis.axvline(float(row["words"][-1]["end"]), color="#b23a48", linewidth=0.45)
            for syllable_index, syllable in enumerate(row["syllables"]):
                start_time = float(syllable["start"])
                end_time = float(syllable["end"])
                axis.axvline(start_time, color="#2a7f62", linewidth=0.35, alpha=0.65, linestyle=":")
                axis.text(
                    (start_time + end_time) / 2,
                    0.02 if syllable_index % 2 == 0 else 0.16,
                    str(syllable["label"]),
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=4.2,
                    rotation=55,
                    color="#155c43",
                )
            axis.axvline(
                float(row["syllables"][-1]["end"]),
                color="#2a7f62",
                linewidth=0.35,
                alpha=0.65,
                linestyle=":",
            )
            axis.set_xlim(0, float(row["duration"]))
            axis.set_ylim(-1.0, 1.0)
            axis.set_title(
                f"{row['id']} | {len(row['words'])} words | "
                f"{len(row['phone_intervals'])} phones | {len(row['syllables'])} syllables",
                loc="left",
                fontsize=9,
            )
            axis.set_xlabel("seconds")
            axis.set_ylabel("amplitude")
        output = output_dir / f"stage2_review_sheet_{sheet_index}.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    for output in render_review_sheets(args.manifest, args.output_dir, args.count):
        print(output)


if __name__ == "__main__":
    main()
