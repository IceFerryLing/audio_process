from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.data.normalization import (  # noqa: E402
    NORMALIZER_VERSION,
    NormalizationError,
    normalize_librispeech_text,
)


class NormalizationTests(unittest.TestCase):
    def test_normalizes_case_punctuation_and_spacing(self) -> None:
        result = normalize_librispeech_text("  Hello,   world! ")
        self.assertEqual(result.normalized, "HELLO WORLD")
        self.assertEqual(result.words, ("HELLO", "WORLD"))
        self.assertEqual(result.version, NORMALIZER_VERSION)

    def test_preserves_internal_apostrophe(self) -> None:
        result = normalize_librispeech_text("Marguerite\u2019s life")
        self.assertEqual(result.normalized, "MARGUERITE'S LIFE")

    def test_splits_hyphenated_words_deterministically(self) -> None:
        result = normalize_librispeech_text("goal-directed")
        self.assertEqual(result.words, ("GOAL", "DIRECTED"))

    def test_rejects_digits_without_expansion_policy(self) -> None:
        with self.assertRaisesRegex(NormalizationError, "digits"):
            normalize_librispeech_text("chapter 16")

    def test_rejects_empty_nonlexical_text(self) -> None:
        with self.assertRaises(NormalizationError):
            normalize_librispeech_text("... -- ...")


if __name__ == "__main__":
    unittest.main()

