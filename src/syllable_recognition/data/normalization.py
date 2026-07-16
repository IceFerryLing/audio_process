"""Versioned normalization for LibriSpeech-style English transcripts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


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
    """Normalize text while rejecting digits that need an expansion policy."""
    if not isinstance(text, str) or not text.strip():
        raise NormalizationError("text must be a non-empty string")
    canonical = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).upper()
    if _DIGIT_RE.search(canonical):
        raise NormalizationError("digits require an explicit expansion and are rejected in v1")
    canonical = canonical.replace("-", " ")
    canonical = _NON_LEXICAL_RE.sub(" ", canonical)
    words = tuple(_WORD_RE.findall(canonical))
    normalized = " ".join(words)
    if not words:
        raise NormalizationError("text contains no lexical English words")
    return NormalizedText(original=text, normalized=normalized, words=words)

