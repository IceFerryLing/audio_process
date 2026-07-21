from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.data.pronunciation import PronunciationResolver  # noqa: E402
from syllable_recognition.data.syllabification import Syllabifier  # noqa: E402
from syllable_recognition.data.transcription import (  # noqa: E402
    ARPABET_TRANSCRIPTION_VERSION,
    IPA_TRANSCRIPTION_VERSION,
    PhoneticTranscriber,
    transcribe_arpabet,
    transcribe_ipa,
)


class TranscriptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = PronunciationResolver(
            {
                "HELLO": [["HH", "AH0", "L", "OW1"]],
                "WORLD": [["W", "ER1", "L", "D"]],
                "ABOUT": [["AH0", "B", "AW1", "T"]],
                "SECONDARY": [["S", "EH1", "K", "AH0", "N", "D", "EH2", "R", "IY0"]],
            },
            lambda word: ["T", "EH1", "S", "T"],
            dictionary_name="test-dictionary",
            g2p_name="test-g2p",
        )
        self.syllabifier = Syllabifier(
            "test-word-internal-v1",
            {
                ("B",),
                ("D",),
                ("L",),
                ("N",),
                ("R",),
            },
        )
        self.transcriber = PhoneticTranscriber(self.resolver, self.syllabifier)

    def test_arpabet_interface_preserves_training_labels(self) -> None:
        result = self.transcriber.transcribe_arpabet("Hello, world!")

        self.assertEqual(result["schema_version"], ARPABET_TRANSCRIPTION_VERSION)
        self.assertEqual(result["alphabet"], "arpabet")
        self.assertEqual(result["text"], "HELLO WORLD")
        self.assertEqual(result["words"][0]["phones"], ["HH", "AH0", "L", "OW1"])
        self.assertEqual(
            [syllable["label"] for syllable in result["words"][0]["syllables"]],
            ["HH-AH0", "L-OW1"],
        )

    def test_ipa_interface_is_derived_from_the_same_syllables(self) -> None:
        result = self.transcriber.transcribe_ipa("Hello, world!")

        self.assertEqual(result["schema_version"], IPA_TRANSCRIPTION_VERSION)
        self.assertEqual(result["alphabet"], "ipa")
        self.assertEqual(result["dialect"], "general-american-broad")
        self.assertEqual(result["words"][0]["label"], "həˈloʊ")
        self.assertEqual(result["words"][1]["label"], "ˈwɝld")
        self.assertEqual(
            [syllable["stress"] for syllable in result["words"][0]["syllables"]],
            [0, 1],
        )

    def test_ipa_handles_unstressed_and_secondary_stress(self) -> None:
        about = self.transcriber.transcribe_ipa("ABOUT")
        secondary = self.transcriber.transcribe_ipa("SECONDARY")

        self.assertEqual(about["words"][0]["label"], "əˈbaʊt")
        self.assertEqual(secondary["words"][0]["label"], "ˈsɛkənˌdɛɹi")

    def test_function_interfaces_match_class_interfaces(self) -> None:
        arpabet = transcribe_arpabet(
            "HELLO",
            resolver=self.resolver,
            syllabifier=self.syllabifier,
        )
        ipa = transcribe_ipa(
            "HELLO",
            resolver=self.resolver,
            syllabifier=self.syllabifier,
        )

        self.assertEqual(arpabet, self.transcriber.transcribe_arpabet("HELLO"))
        self.assertEqual(ipa, self.transcriber.transcribe_ipa("HELLO"))


if __name__ == "__main__":
    unittest.main()
