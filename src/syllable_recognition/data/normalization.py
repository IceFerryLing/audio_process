"""Versioned normalization for LibriSpeech-style English transcripts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# 规范化规则会影响词典查询和所有下游标签，因此必须显式版本化。
NORMALIZER_VERSION = "librispeech-english-v1"
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})
_WORD_RE = re.compile(r"[A-Z]+(?:'[A-Z]+)*")
_DIGIT_RE = re.compile(r"\d")
_NON_LEXICAL_RE = re.compile(r"[^A-Z']+")


class NormalizationError(ValueError):
    """Raised when text cannot be normalized without an undocumented choice."""


@dataclass(frozen=True)
class NormalizedText:
    original: str
    normalized: str
    words: tuple[str, ...]
    version: str = NORMALIZER_VERSION


def normalize_librispeech_text(text: str) -> NormalizedText:
    """把 LibriSpeech 风格文本转换成可复现的词序列。

    v1 故意拒绝数字，因为 ``123`` 可能读作逐位数字，也可能读作整数。
    在没有版本化展开策略时静默选择一种读法，会让音频和标签错位。
    """
    if not isinstance(text, str) or not text.strip():
        raise NormalizationError("text must be a non-empty string")
    # NFKC 先统一全角字符等 Unicode 变体，再统一不同形态的英文撇号。
    canonical = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).upper()
    if _DIGIT_RE.search(canonical):
        raise NormalizationError("digits require an explicit expansion and are rejected in v1")
    # 连字符在当前规则中被视为词边界；其他非词汇字符统一压缩为空格。
    canonical = canonical.replace("-", " ")
    canonical = _NON_LEXICAL_RE.sub(" ", canonical)
    words = tuple(_WORD_RE.findall(canonical))
    normalized = " ".join(words)
    if not words:
        raise NormalizationError("text contains no lexical English words")
    return NormalizedText(original=text, normalized=normalized, words=words)

