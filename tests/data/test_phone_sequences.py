from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.data.phone_sequences import (  # noqa: E402
    PhoneVocabulary,
    convolution_output_length,
    minimum_ctc_frames,
)


class PhoneSequenceTests(unittest.TestCase):
    def test_vocabulary_is_deterministic_and_train_only(self) -> None:
        vocabulary = PhoneVocabulary.from_training_sequences([["HH", "AH0"], ["AH0", "Z"]])
        self.assertEqual(vocabulary.tokens, ("<blank>", "<unk>", "AH0", "HH", "Z"))
        self.assertEqual(vocabulary.encode(["HH", "MISSING"]), [3, 1])

    def test_minimum_ctc_frames_accounts_for_adjacent_repeats(self) -> None:
        self.assertEqual(minimum_ctc_frames(["AH0", "AH0", "T"]), 4)
        self.assertEqual(minimum_ctc_frames(["AH0", "T", "AH0"]), 3)

    def test_hubert_convolution_length_for_one_second(self) -> None:
        kernels = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
        self.assertEqual(convolution_output_length(16_000, kernels, strides), 49)

    def test_invalid_convolution_spec_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            convolution_output_length(16_000, [10], [5, 2])


if __name__ == "__main__":
    unittest.main()
