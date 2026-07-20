from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.decoding.phone_ctc import (  # noqa: E402
    collapse_ctc_ids,
    forced_align_ctc,
    frame_interval_to_seconds,
)
from syllable_recognition.metrics.per import phone_edit_distance, phone_error_rate  # noqa: E402


class PhoneCTCDecodingTests(unittest.TestCase):
    def test_collapse_preserves_repeat_separated_by_blank(self) -> None:
        self.assertEqual(collapse_ctc_ids([0, 2, 2, 0, 2, 3, 3], blank_id=0), [2, 2, 3])

    def test_phone_error_rate_uses_corpus_denominator(self) -> None:
        references = [["HH", "AH0"], ["W", "ER1", "L", "D"]]
        hypotheses = [["HH", "AE0"], ["W", "ER1", "D"]]
        self.assertEqual(phone_edit_distance(references[0], hypotheses[0]), 1)
        self.assertAlmostEqual(phone_error_rate(references, hypotheses), 2 / 6)

    def test_empty_reference_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            phone_error_rate([], [])

    def test_forced_alignment_consumes_target_in_order(self) -> None:
        probabilities = torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
                [0.1, 0.9, 0.0],
                [0.9, 0.05, 0.05],
                [0.1, 0.0, 0.9],
                [0.1, 0.0, 0.9],
                [0.9, 0.05, 0.05],
            ],
            dtype=torch.float32,
        )
        aligned = forced_align_ctc(probabilities.clamp_min(1e-6).log(), [1, 2], blank_id=0)
        self.assertEqual([(item.start_frame, item.end_frame) for item in aligned], [(1, 3), (4, 6)])

    def test_repeated_target_requires_intervening_blank(self) -> None:
        probabilities = torch.tensor(
            [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
            dtype=torch.float32,
        )
        aligned = forced_align_ctc(probabilities.log(), [1, 1], blank_id=0)
        self.assertEqual([(item.start_frame, item.end_frame) for item in aligned], [(0, 1), (2, 3)])

    def test_frame_interval_conversion_is_bounded(self) -> None:
        self.assertEqual(
            frame_interval_to_seconds(
                10,
                12,
                sample_rate=16_000,
                frame_hop_samples=320,
                audio_duration=1.0,
            ),
            (0.2, 0.24),
        )


if __name__ == "__main__":
    unittest.main()
