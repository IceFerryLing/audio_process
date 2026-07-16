"""Parse MFA TextGrids and attach authoritative phone/syllable timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praatio import textgrid

from syllable_recognition.contracts.guided import CONSONANTS, VOWELS


class AlignmentParseError(ValueError):
    """Raised when MFA output cannot be matched exactly to prepared labels."""


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class ParsedAlignment:
    words: tuple[Interval, ...]
    phones: tuple[Interval, ...]


def _normalize_phone(label: str) -> str:
    phone = label.strip().upper()
    if phone[-1:] in {"0", "1", "2"}:
        if phone[:-1] not in VOWELS:
            raise AlignmentParseError(f"invalid stressed MFA phone {label!r}")
    elif phone not in CONSONANTS:
        raise AlignmentParseError(f"unknown MFA phone {label!r}")
    return phone


def parse_mfa_textgrid(
    path: Path,
    *,
    word_tier: str = "words",
    phone_tier: str = "phones",
    silence_phones: set[str] | None = None,
) -> ParsedAlignment:
    """Read the configured tiers without silently accepting unknown labels."""
    silence = {item.upper() for item in (silence_phones or {"sil", "sp"})}
    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True, reportingMode="error")
    if word_tier not in grid.tierNames or phone_tier not in grid.tierNames:
        raise AlignmentParseError(f"{path.name} is missing {word_tier!r} or {phone_tier!r} tier")

    words = tuple(
        Interval(float(entry.start), float(entry.end), entry.label.strip().upper())
        for entry in grid.getTier(word_tier).entries
        if entry.label.strip()
    )
    phones: list[Interval] = []
    for entry in grid.getTier(phone_tier).entries:
        label = entry.label.strip()
        if not label or label.upper() in silence:
            continue
        phones.append(Interval(float(entry.start), float(entry.end), _normalize_phone(label)))
    return ParsedAlignment(words, tuple(phones))


def align_prepared_row(
    row: dict[str, Any],
    textgrid_path: Path,
    *,
    word_tier: str,
    phone_tier: str,
    silence_phones: set[str],
) -> dict[str, Any]:
    parsed = parse_mfa_textgrid(
        textgrid_path,
        word_tier=word_tier,
        phone_tier=phone_tier,
        silence_phones=silence_phones,
    )
    expected_words = row["words"]
    if len(parsed.words) != len(expected_words):
        raise AlignmentParseError(
            f"word count mismatch: expected {len(expected_words)}, found {len(parsed.words)}"
        )

    timed_words: list[dict[str, Any]] = []
    timed_syllables: list[dict[str, Any]] = []
    flat_phone_intervals: list[dict[str, Any]] = []
    previous_syllable_end = 0.0
    epsilon = 1e-5

    for word_index, (word_interval, expected_word) in enumerate(zip(parsed.words, expected_words)):
        if word_interval.label != expected_word["word"]:
            raise AlignmentParseError(
                f"word {word_index} mismatch: expected {expected_word['word']!r}, found {word_interval.label!r}"
            )
        word_phones = [
            phone
            for phone in parsed.phones
            if phone.start >= word_interval.start - epsilon and phone.end <= word_interval.end + epsilon
        ]
        actual_phones = [phone.label for phone in word_phones]
        if actual_phones != expected_word["phones"]:
            raise AlignmentParseError(
                f"{word_interval.label} phone mismatch: expected={expected_word['phones']!r}, actual={actual_phones!r}"
            )

        phone_cursor = 0
        word_syllables: list[dict[str, Any]] = []
        for syllable_index, syllable in enumerate(expected_word["syllables"]):
            phone_count = len(syllable["phones"])
            intervals = word_phones[phone_cursor : phone_cursor + phone_count]
            if len(intervals) != phone_count:
                raise AlignmentParseError(f"{word_interval.label} syllable phone intervals are incomplete")
            start, end = intervals[0].start, intervals[-1].end
            if end <= start or start < previous_syllable_end - epsilon:
                raise AlignmentParseError("syllable timestamps are non-monotonic or overlapping")
            if start < -epsilon or end > float(row["duration"]) + epsilon:
                raise AlignmentParseError("syllable timestamp is outside the audio duration")
            timed = {
                **syllable,
                "word_index": word_index,
                "syllable_index": syllable_index,
                "start": round(start, 6),
                "end": round(end, 6),
            }
            timed_syllables.append(timed)
            word_syllables.append(timed)
            previous_syllable_end = end
            phone_cursor += phone_count

        if phone_cursor != len(word_phones):
            raise AlignmentParseError(f"{word_interval.label} has unassigned phone intervals")
        timed_words.append(
            {
                **expected_word,
                "start": round(word_interval.start, 6),
                "end": round(word_interval.end, 6),
                "syllables": word_syllables,
            }
        )
        flat_phone_intervals.extend(
            {"phone": phone.label, "start": round(phone.start, 6), "end": round(phone.end, 6)}
            for phone in word_phones
        )

    if len(flat_phone_intervals) != len(row["phones"]):
        raise AlignmentParseError("not every target phone received an interval")
    return {
        **row,
        "alignment_status": "aligned",
        "alignment_source": "mfa",
        "mfa_textgrid": textgrid_path.as_posix(),
        "words": timed_words,
        "phone_intervals": flat_phone_intervals,
        "syllables": timed_syllables,
    }

