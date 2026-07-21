from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.data.librispeech_download import (  # noqa: E402
    DatasetSource,
    LibriSpeechDownloadError,
    SelectionState,
    SplitDownloadSpec,
    _discover_split_shards,
    load_librispeech_download_config,
)


class SelectionStateTests(unittest.TestCase):
    def test_hour_target_respects_speaker_cap_and_minimum_speakers(self) -> None:
        spec = SplitDownloadSpec(
            name="train-clean-100",
            source_split="train.clean.100",
            source_shards=("all/train.clean.100/0000.parquet",),
            output_dir=Path("unused"),
            enabled=True,
            target_hours=4.0 / 3600.0,
            target_items=None,
            max_seconds_per_speaker=2.5,
            minimum_speakers=2,
        )
        state = SelectionState(spec)

        self.assertTrue(state.can_accept("speaker-a", 2.0))
        state.accept("speaker-a", "chapter-a", 2.0)
        self.assertFalse(state.can_accept("speaker-a", 1.0))
        state.accept("speaker-b", "chapter-b", 2.0)

        self.assertTrue(state.complete)
        state.validate_complete()
        self.assertEqual(state.selected_items, 2)
        self.assertEqual(len(state.speaker_seconds), 2)

    def test_item_target_fails_when_records_are_exhausted(self) -> None:
        spec = SplitDownloadSpec(
            name="dev-clean",
            source_split="dev.clean",
            source_shards=(),
            output_dir=Path("unused"),
            enabled=True,
            target_hours=None,
            target_items=2,
            max_seconds_per_speaker=None,
            minimum_speakers=1,
        )
        state = SelectionState(spec)
        state.accept("speaker-a", "chapter-a", 1.0)

        with self.assertRaisesRegex(LibriSpeechDownloadError, "requested 2 items"):
            state.validate_complete()

    def test_explicit_shards_skip_remote_discovery(self) -> None:
        source = DatasetSource("dataset", "all", "revision", 16_000)
        spec = SplitDownloadSpec(
            name="train-clean-100",
            source_split="train.clean.100",
            source_shards=("all/train.clean.100/0000.parquet",),
            output_dir=Path("unused"),
            enabled=True,
            target_hours=1.0,
            target_items=None,
            max_seconds_per_speaker=240.0,
            minimum_speakers=10,
        )

        self.assertEqual(
            _discover_split_shards(source, spec),
            ["all/train.clean.100/0000.parquet"],
        )


class DownloadConfigTests(unittest.TestCase):
    def test_loads_stage3_config_and_resolves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "configs" / "data"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "download.yaml"
            payload = {
                "stage": "3-librispeech-download",
                "mode": "guided",
                "dataset": {
                    "id": "openslr/librispeech_asr",
                    "config": "all",
                    "revision": "fixed-revision",
                    "expected_sample_rate": 16000,
                },
                "splits": {
                    "train-clean-100": {
                        "source_split": "train.clean.100",
                        "source_shards": ["all/train.clean.100/0000.parquet"],
                        "output_dir": "data/librispeech/train-clean-100",
                        "enabled": True,
                        "target_hours": 1.0,
                        "target_items": None,
                        "max_seconds_per_speaker": 240.0,
                        "minimum_speakers": 10,
                    }
                },
                "output": {
                    "report": "artifacts/reports/download.json",
                    "resolved_config": "artifacts/reports/download_config.yaml",
                },
            }
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            config = load_librispeech_download_config(config_path)

            self.assertEqual(config.source.expected_sample_rate, 16_000)
            self.assertEqual(config.splits[0].target_hours, 1.0)
            self.assertEqual(
                config.splits[0].output_dir,
                (root / "data/librispeech/train-clean-100").resolve(),
            )

    def test_rejects_two_subset_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "configs" / "data"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "download.yaml"
            payload = {
                "stage": "3-librispeech-download",
                "mode": "guided",
                "dataset": {
                    "id": "dataset",
                    "config": "all",
                    "revision": "revision",
                    "expected_sample_rate": 16000,
                },
                "splits": {
                    "train-clean-100": {
                        "source_split": "train.clean.100",
                        "output_dir": "data/train",
                        "target_hours": 1.0,
                        "target_items": 10,
                    }
                },
                "output": {
                    "report": "artifacts/report.json",
                    "resolved_config": "artifacts/config.yaml",
                },
            }
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            with self.assertRaisesRegex(LibriSpeechDownloadError, "cannot set both"):
                load_librispeech_download_config(config_path)


if __name__ == "__main__":
    unittest.main()
