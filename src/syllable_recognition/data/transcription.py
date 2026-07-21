"""Stable ARPAbet and broad General American IPA text interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from syllable_recognition.contracts.guided import CONSONANTS, VOWELS

from .normalization import normalize_librispeech_text
from .pronunciation import PronunciationResolver
from .syllabification import Syllable, Syllabifier


ARPABET_TRANSCRIPTION_VERSION = "arpabet-transcription-v1"
IPA_TRANSCRIPTION_VERSION = "general-american-broad-ipa-transcription-v1"

# 这是音位级宽式映射，不尝试预测闪音、喉塞音等真实语流中的音位变体。
_CONSONANT_IPA = {
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

_VOWEL_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
}

_STRESS_MARK = {0: "", 1: "ˈ", 2: "ˌ"}


def _vowel_to_ipa(nucleus: str, stress: int) -> str:
    """映射元音；AH/ER 的宽式美式 IPA 取决于是否重读。"""
    if nucleus == "AH":
        return "ə" if stress == 0 else "ʌ"
    if nucleus == "ER":
        return "ɚ" if stress == 0 else "ɝ"
    return _VOWEL_IPA[nucleus]


def _syllable_to_ipa(syllable: Syllable) -> dict[str, Any]:
    onset = tuple(_CONSONANT_IPA[phone] for phone in syllable.onset)
    nucleus = _vowel_to_ipa(syllable.nucleus, syllable.stress)
    coda = tuple(_CONSONANT_IPA[phone] for phone in syllable.coda)
    phones = onset + (nucleus,) + coda
    return {
        "phones": list(phones),
        # IPA 重音符号放在音节开头，而不是附着到元音 token 上。
        "label": f"{_STRESS_MARK[syllable.stress]}{''.join(phones)}",
        "onset": list(onset),
        "nucleus": nucleus,
        "coda": list(coda),
        "stress": syllable.stress,
    }


class PhoneticTranscriber:
    """共享同一规范化、发音选择和音节化结果的双版本转写器。"""

    def __init__(self, resolver: PronunciationResolver, syllabifier: Syllabifier) -> None:
        self.resolver = resolver
        self.syllabifier = syllabifier

    @classmethod
    def from_default_packages(cls, syllabifier_config: Path) -> "PhoneticTranscriber":
        """使用锁定的 CMUdict/G2P 包和指定版本音节规则构造转写器。"""
        return cls(
            PronunciationResolver.from_default_packages(),
            Syllabifier.from_config(syllabifier_config),
        )

    def transcribe_arpabet(self, text: str) -> dict[str, Any]:
        """返回权威 ARPAbet 视图，适合 manifest、训练和对齐。"""
        normalized = normalize_librispeech_text(text)
        pronunciations = self.resolver.resolve_many(normalized.words)
        words: list[dict[str, Any]] = []
        for pronunciation in pronunciations:
            syllables = self.syllabifier.syllabify(pronunciation.phones)
            words.append(
                {
                    "word": pronunciation.word,
                    "phones": list(pronunciation.phones),
                    "label": "-".join(pronunciation.phones),
                    "syllables": [syllable.to_dict() for syllable in syllables],
                    "source": pronunciation.source,
                    "variant_index": pronunciation.variant_index,
                    "variant_count": pronunciation.variant_count,
                }
            )
        return {
            "schema_version": ARPABET_TRANSCRIPTION_VERSION,
            "alphabet": "arpabet",
            "text": normalized.normalized,
            "normalizer_version": normalized.version,
            "syllabifier_version": self.syllabifier.rule_version,
            "words": words,
        }

    def transcribe_ipa(self, text: str) -> dict[str, Any]:
        """返回由权威 ARPAbet 结果派生的宽式美式 IPA 显示视图。"""
        arpabet = self.transcribe_arpabet(text)
        words: list[dict[str, Any]] = []
        for word in arpabet["words"]:
            arpabet_syllables = [
                Syllable(
                    onset=tuple(syllable["onset"]),
                    nucleus=syllable["nucleus"],
                    coda=tuple(syllable["coda"]),
                    stress=int(syllable["stress"]),
                )
                for syllable in word["syllables"]
            ]
            ipa_syllables = [_syllable_to_ipa(syllable) for syllable in arpabet_syllables]
            words.append(
                {
                    "word": word["word"],
                    "phones": [
                        phone
                        for syllable in ipa_syllables
                        for phone in syllable["phones"]
                    ],
                    "label": "".join(syllable["label"] for syllable in ipa_syllables),
                    "syllables": ipa_syllables,
                    "source": f"arpabet-derived:{word['source']}",
                    "variant_index": word["variant_index"],
                    "variant_count": word["variant_count"],
                }
            )
        return {
            "schema_version": IPA_TRANSCRIPTION_VERSION,
            "alphabet": "ipa",
            "dialect": "general-american-broad",
            "text": arpabet["text"],
            "normalizer_version": arpabet["normalizer_version"],
            "syllabifier_version": arpabet["syllabifier_version"],
            "source_alphabet": "arpabet",
            "words": words,
        }


def transcribe_arpabet(
    text: str,
    *,
    resolver: PronunciationResolver,
    syllabifier: Syllabifier,
) -> dict[str, Any]:
    """函数式 ARPAbet 接口，便于 CLI、测试或上层服务直接调用。"""
    return PhoneticTranscriber(resolver, syllabifier).transcribe_arpabet(text)


def transcribe_ipa(
    text: str,
    *,
    resolver: PronunciationResolver,
    syllabifier: Syllabifier,
) -> dict[str, Any]:
    """函数式宽式美式 IPA 接口；内部仍以 ARPAbet 为权威标签。"""
    return PhoneticTranscriber(resolver, syllabifier).transcribe_ipa(text)


if set(_CONSONANT_IPA) != CONSONANTS:
    raise RuntimeError("IPA consonant mapping does not cover the ARPAbet contract")
if set(_VOWEL_IPA) | {"AH", "ER"} != VOWELS:
    raise RuntimeError("IPA vowel mapping does not cover the ARPAbet contract")


__all__ = [
    "ARPABET_TRANSCRIPTION_VERSION",
    "IPA_TRANSCRIPTION_VERSION",
    "PhoneticTranscriber",
    "transcribe_arpabet",
    "transcribe_ipa",
]
