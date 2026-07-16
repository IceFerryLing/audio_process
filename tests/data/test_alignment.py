from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.data.alignment import (  # noqa: E402
    AlignmentParseError,
    align_prepared_row,
    parse_mfa_textgrid,
)


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "tests/fixtures/data/hello_world.TextGrid"
        cls.row = {
            "id": "sample-001",
            "duration": 1.0,
            "phones": ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"],
            "words": [
                {
                    "word": "HELLO",
                    "phones": ["HH", "AH0", "L", "OW1"],
                    "source": "test",
                    "variant_index": 0,
                    "variant_count": 1,
                    "syllables": [
                        {"phones": ["HH", "AH0"], "label": "HH-AH0", "onset": ["HH"], "nucleus": "AH", "coda": [], "stress": 0},
                        {"phones": ["L", "OW1"], "label": "L-OW1", "onset": ["L"], "nucleus": "OW", "coda": [], "stress": 1},
                    ],
                },
                {
                    "word": "WORLD",
                    "phones": ["W", "ER1", "L", "D"],
                    "source": "test",
                    "variant_index": 0,
                    "variant_count": 1,
                    "syllables": [
                        {"phones": ["W", "ER1", "L", "D"], "label": "W-ER1-L-D", "onset": ["W"], "nucleus": "ER", "coda": ["L", "D"], "stress": 1}
                    ],
                },
            ],
            "syllables": [],
        }

    def test_parses_tiers_and_filters_configured_silence(self) -> None:
        result = parse_mfa_textgrid(self.path)
        self.assertEqual([word.label for word in result.words], ["HELLO", "WORLD"])
        self.assertEqual(len(result.phones), 8)

    def test_attaches_word_phone_and_syllable_timestamps(self) -> None:
        result = align_prepared_row(
            self.row,
            self.path,
            word_tier="words",
            phone_tier="phones",
            silence_phones={"sil", "sp"},
        )
        self.assertEqual(result["alignment_status"], "aligned")
        self.assertEqual([item["label"] for item in result["syllables"]], ["HH-AH0", "L-OW1", "W-ER1-L-D"])
        self.assertEqual((result["syllables"][0]["start"], result["syllables"][0]["end"]), (0.1, 0.29))
        self.assertEqual(len(result["phone_intervals"]), len(self.row["phones"]))

    def test_rejects_phone_mismatch(self) -> None:
        row = copy.deepcopy(self.row)
        row["words"][0]["phones"][0] = "K"
        with self.assertRaisesRegex(AlignmentParseError, "phone mismatch"):
            align_prepared_row(
                row,
                self.path,
                word_tier="words",
                phone_tier="phones",
                silence_phones={"sil", "sp"},
            )


if __name__ == "__main__":
    unittest.main()

