from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.data.pronunciation import (  # noqa: E402
    PronunciationError,
    PronunciationResolver,
)


class PronunciationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.g2p_calls: list[str] = []

        def g2p(word: str) -> list[str]:
            self.g2p_calls.append(word)
            return ["T", "EH1", "S", "T"]

        self.resolver = PronunciationResolver(
            {
                "HELLO": [["HH", "AH0", "L", "OW1"], ["HH", "EH1", "L", "OW0"]],
                "NAME": [["N", "EY1", "M"]],
                "MISTRESS": [["M", "IH1", "S", "T", "R", "AH0", "S"]],
                "CAT": [["K", "AE1", "T"]],
                "BAD": [["B", "AE"]],
            },
            g2p,
            dictionary_name="test-dictionary",
            g2p_name="test-g2p",
        )

    def test_selects_first_dictionary_variant(self) -> None:
        result = self.resolver.resolve("HELLO")
        self.assertEqual(result.phones, ("HH", "AH0", "L", "OW1"))
        self.assertEqual(result.variant_index, 0)
        self.assertEqual(result.variant_count, 2)
        self.assertEqual(result.source, "test-dictionary")
        self.assertEqual(self.g2p_calls, [])

    def test_oov_uses_explicit_g2p(self) -> None:
        result = self.resolver.resolve("UNKNOWN")
        self.assertEqual(result.phones, ("T", "EH1", "S", "T"))
        self.assertEqual(result.source, "test-g2p")
        self.assertEqual(self.g2p_calls, ["UNKNOWN"])

    def test_possessive_reuses_known_root_before_g2p(self) -> None:
        self.assertEqual(self.resolver.resolve("NAME'S").phones, ("N", "EY1", "M", "Z"))
        self.assertEqual(
            self.resolver.resolve("MISTRESS'S").phones,
            ("M", "IH1", "S", "T", "R", "AH0", "S", "IH0", "Z"),
        )
        self.assertEqual(self.resolver.resolve("CAT'S").phones, ("K", "AE1", "T", "S"))
        self.assertEqual(self.g2p_calls, [])

    def test_rejects_unstressed_vowel(self) -> None:
        with self.assertRaisesRegex(PronunciationError, "missing stress"):
            self.resolver.resolve("BAD")


if __name__ == "__main__":
    unittest.main()
