from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.metrics.alignment import paired_phone_boundary_metrics  # noqa: E402


class AlignmentMetricTests(unittest.TestCase):
    def test_paired_boundary_metrics_apply_tolerances(self) -> None:
        reference = [{"phone": "HH", "start": 0.10, "end": 0.20}]
        hypothesis = [{"phone": "HH", "start": 0.11, "end": 0.26}]
        result = paired_phone_boundary_metrics(reference, hypothesis)
        self.assertAlmostEqual(result["boundary_mae_ms"], 35.0)
        self.assertEqual(result["tolerances_ms"]["20"]["f1"], 0.5)
        self.assertEqual(result["tolerances_ms"]["50"]["f1"], 0.5)

    def test_phone_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_phone_boundary_metrics(
                [{"phone": "HH", "start": 0.1, "end": 0.2}],
                [{"phone": "W", "start": 0.1, "end": 0.2}],
            )


if __name__ == "__main__":
    unittest.main()
