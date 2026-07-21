"""Deterministic CMUdict lookup with explicit OOV G2P fallback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from typing import Any

from syllable_recognition.contracts.guided import CONSONANTS, VOWELS


PronunciationGenerator = Callable[[str], Sequence[str]]


class PronunciationError(ValueError):
    """Raised when a word cannot produce a valid ARPAbet pronunciation."""


@dataclass(frozen=True)
class Pronunciation:
    word: str
    phones: tuple[str, ...]
    source: str
    variant_index: int
    variant_count: int


def _validate_phones(word: str, phones: Sequence[str]) -> tuple[str, ...]:
    """校验一个发音是否符合项目固定的 ARPAbet 契约。"""
    result = tuple(phone.upper() for phone in phones)
    if not result:
        raise PronunciationError(f"{word!r} produced no phones")
    vowel_count = 0
    for phone in result:
        # 元音必须携带 0/1/2 重音；辅音则不能伪装成带重音音素。
        if phone[-1:] in {"0", "1", "2"}:
            base = phone[:-1]
            if base not in VOWELS:
                raise PronunciationError(f"{word!r} has invalid stressed phone {phone!r}")
            vowel_count += 1
        elif phone not in CONSONANTS:
            if phone in VOWELS:
                raise PronunciationError(f"{word!r} vowel {phone!r} is missing stress")
            raise PronunciationError(f"{word!r} has unknown phone {phone!r}")
    if vowel_count == 0:
        raise PronunciationError(f"{word!r} pronunciation has no vowel nucleus")
    return result


class PronunciationResolver:
    """为每个规范化单词确定一个可复现的 ARPAbet 发音。

    当前基线优先使用 CMUdict 的第一个变体，词典未收录时才调用 G2P。
    这是确定性基线，不表示第一个变体一定与每条真实录音完全一致。
    """

    def __init__(
        self,
        lexicon: Mapping[str, Sequence[Sequence[str]]],
        g2p: PronunciationGenerator,
        *,
        dictionary_name: str = "cmudict",
        g2p_name: str = "g2p_en",
    ) -> None:
        self._lexicon = {key.upper(): value for key, value in lexicon.items()}
        self._g2p = g2p
        self.dictionary_name = dictionary_name
        self.g2p_name = g2p_name

    @classmethod
    def from_default_packages(cls) -> "PronunciationResolver":
        import cmudict
        import nltk

        original_download = nltk.download
        nltk.download = lambda *args, **kwargs: False
        try:
            g2p_module = import_module("g2p_en.g2p")
        finally:
            nltk.download = original_download

        raw_lexicon: dict[str, Any] = cmudict.dict()
        # g2p_en 默认依赖 NLTK 单独下载的 CMUdict。这里复用锁定版本的
        # cmudict Python 包，避免构造 resolver 时发生不可见的联网下载。
        g2p_module.cmudict = cmudict
        generator = g2p_module.G2p()

        def generate(word: str) -> Sequence[str]:
            return generator.predict(word.lower())

        return cls(
            raw_lexicon,
            generate,
            dictionary_name=f"cmudict-{version('cmudict')}",
            g2p_name=f"g2p_en-{version('g2p-en')}",
        )

    def resolve(self, word: str) -> Pronunciation:
        """解析一个单词，并保留词典/G2P 来源和发音变体信息。"""
        variants = self._lexicon.get(word.upper())
        if variants:
            # v1 固定选择第一个变体以保证重跑一致，后续可升级为声学选变体。
            phones = _validate_phones(word, variants[0])
            return Pronunciation(word, phones, self.dictionary_name, 0, len(variants))
        if word.upper().endswith("'S"):
            # 所有格 'S 根据词尾清浊和嘶擦属性规则化为 /S/、/Z/ 或 /IH0 Z/。
            base_word = word.upper()[:-2]
            base_variants = self._lexicon.get(base_word)
            if base_variants:
                base_phones = _validate_phones(base_word, base_variants[0])
                final_phone = base_phones[-1].rstrip("012")
                if final_phone in {"S", "Z", "SH", "ZH", "CH", "JH"}:
                    suffix = ("IH0", "Z")
                elif final_phone in {"P", "T", "K", "F", "TH"}:
                    suffix = ("S",)
                else:
                    suffix = ("Z",)
                return Pronunciation(
                    word,
                    _validate_phones(word, base_phones + suffix),
                    f"{self.dictionary_name}+possessive-s-v1",
                    0,
                    len(base_variants),
                )
        # OOV 不能静默删除，否则音素标签会与录音中的单词顺序错位。
        phones = _validate_phones(word, self._g2p(word))
        return Pronunciation(word, phones, self.g2p_name, 0, 1)

    def resolve_many(self, words: Sequence[str]) -> tuple[Pronunciation, ...]:
        return tuple(self.resolve(word) for word in words)
