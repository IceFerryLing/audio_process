from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.data.syllabification import (  # noqa: E402
    SyllabificationError,
    Syllabifier,
)


class SyllabificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.syllabifier = Syllabifier.from_config(
            ROOT / "configs/data/syllabifier_arpabet_v1.yaml"
        )

    def test_hello(self) -> None:
        syllables = self.syllabifier.syllabify(["HH", "AH0", "L", "OW1"])
        self.assertEqual([item.label for item in syllables], ["HH-AH0", "L-OW1"])
        self.assertEqual(syllables[0].coda, ())
        self.assertEqual(syllables[1].onset, ("L",))

    def test_world(self) -> None:
        syllables = self.syllabifier.syllabify(["W", "ER1", "L", "D"])
        self.assertEqual(len(syllables), 1)
        self.assertEqual(syllables[0].to_dict()["coda"], ["L", "D"])

    def test_longest_legal_onset_suffix(self) -> None:
        syllables = self.syllabifier.syllabify(["AE1", "K", "S", "T", "R", "AH0"])
        self.assertEqual(syllables[0].coda, ("K",))
        self.assertEqual(syllables[1].onset, ("S", "T", "R"))

    def test_ng_is_not_forced_into_onset(self) -> None:
        syllables = self.syllabifier.syllabify(["S", "IH1", "NG", "ER0"])
        self.assertEqual(syllables[0].coda, ("NG",))
        self.assertEqual(syllables[1].onset, ())

    def test_adjacent_vowels_create_separate_nuclei(self) -> None:
        syllables = self.syllabifier.syllabify(["IY0", "AA1"])
        self.assertEqual([item.label for item in syllables], ["IY0", "AA1"])

    def test_rejects_missing_nucleus(self) -> None:
        with self.assertRaisesRegex(SyllabificationError, "no vowel nucleus"):
            self.syllabifier.syllabify(["S", "T", "R"])


if __name__ == "__main__":
    unittest.main()

