from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.data.stage2 import summarize_manual_reviews  # noqa: E402


class ManualReviewGateTests(unittest.TestCase):
    def test_only_explicit_passes_open_the_gate(self) -> None:
        reviews = [
            {"id": "a", "review_status": "passed_visual_review"},
            {"id": "b", "review_status": "passed_visual_review"},
        ]
        result = summarize_manual_reviews(reviews, aligned_ids={"a", "b"}, minimum_review=2)
        self.assertTrue(result["manual_gate_passed"])

    def test_pending_review_does_not_open_the_gate(self) -> None:
        reviews = [
            {"id": "a", "review_status": "passed_visual_review"},
            {"id": "b", "review_status": "pending_visual_review"},
        ]
        result = summarize_manual_reviews(reviews, aligned_ids={"a", "b"}, minimum_review=1)
        self.assertFalse(result["manual_gate_passed"])
        self.assertEqual(result["pending_ids"], ["b"])

    def test_failed_review_does_not_open_the_gate(self) -> None:
        reviews = [
            {"id": "a", "review_status": "passed_visual_review"},
            {"id": "b", "review_status": "failed_visual_review"},
        ]
        result = summarize_manual_reviews(reviews, aligned_ids={"a", "b"}, minimum_review=1)
        self.assertFalse(result["manual_gate_passed"])
        self.assertEqual(result["failed_ids"], ["b"])

    def test_invalid_status_and_unknown_id_are_reported(self) -> None:
        reviews = [{"id": "unknown", "review_status": "reviewed"}]
        result = summarize_manual_reviews(reviews, aligned_ids={"a"}, minimum_review=1)
        self.assertFalse(result["manual_gate_passed"])
        self.assertEqual(len(result["review_problems"]), 2)


if __name__ == "__main__":
    unittest.main()
