"""Semantic validation for the Guided MVP output contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


SCHEMA_VERSION = "guided-syllable-result-v1"
VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)
CONSONANTS = frozenset(
    {
        "B",
        "CH",
        "D",
        "DH",
        "F",
        "G",
        "HH",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    }
)

_TOP_LEVEL_KEYS = {"schema_version", "audio_id", "mode", "target_text", "alignment", "segments"}
_SEGMENT_KEYS = {
    "target_index",
    "start",
    "end",
    "phones",
    "label",
    "onset",
    "nucleus",
    "coda",
    "stress",
    "token_id",
    "confidence",
}
_ALIGNMENT_KEYS = {"status", "target_syllable_count", "aligned_syllable_count", "events"}
_EVENT_KEYS = {"type", "target_index", "start", "end", "phones", "confidence"}
_EVENT_TYPES = {"insertion", "deletion", "repetition", "mismatch"}


class ContractValidationError(ValueError):
    """Raised when a result violates a Guided contract invariant."""


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    keys = set(value)
    missing = expected - keys
    unknown = keys - expected
    if missing or unknown:
        raise ContractValidationError(
            f"{context} keys are invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{context} must be a non-empty string")
    return value


def _require_int(value: Any, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ContractValidationError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _require_number(value: Any, context: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value < minimum:
        raise ContractValidationError(f"{context} must be a number >= {minimum}")
    return float(value)


def _require_confidence(value: Any, context: str) -> float:
    confidence = _require_number(value, context)
    if confidence > 1.0:
        raise ContractValidationError(f"{context} must be <= 1")
    return confidence


def _require_sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractValidationError(f"{context} must be an array")
    return value


def _split_phone(phone: Any, context: str) -> tuple[str, int | None]:
    if not isinstance(phone, str) or not phone:
        raise ContractValidationError(f"{context} must be a non-empty ARPAbet phone")
    if phone[-1:] in {"0", "1", "2"}:
        base, stress = phone[:-1], int(phone[-1])
    else:
        base, stress = phone, None
    if base in VOWELS:
        if stress is None:
            raise ContractValidationError(f"{context} vowel {phone!r} must carry stress")
    elif base in CONSONANTS:
        if stress is not None:
            raise ContractValidationError(f"{context} consonant {phone!r} cannot carry stress")
    else:
        raise ContractValidationError(f"{context} contains unknown ARPAbet phone {phone!r}")
    return base, stress


def _require_phone_list(value: Any, context: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    items = _require_sequence(value, context)
    if not allow_empty and not items:
        raise ContractValidationError(f"{context} cannot be empty")
    phones = tuple(items)
    for index, phone in enumerate(phones):
        _split_phone(phone, f"{context}[{index}]")
    return phones


@dataclass(frozen=True)
class SyllableSegment:
    """A target syllable localized by Guided acoustic alignment."""

    target_index: int
    start: float
    end: float
    phones: tuple[str, ...]
    label: str
    onset: tuple[str, ...]
    nucleus: str
    coda: tuple[str, ...]
    stress: int
    token_id: int | None
    confidence: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SyllableSegment":
        _require_exact_keys(value, _SEGMENT_KEYS, "segment")
        target_index = _require_int(value["target_index"], "segment.target_index")
        start = _require_number(value["start"], "segment.start")
        end = _require_number(value["end"], "segment.end")
        if end <= start:
            raise ContractValidationError("segment.end must be greater than segment.start")

        phones = _require_phone_list(value["phones"], "segment.phones", allow_empty=False)
        onset = _require_phone_list(value["onset"], "segment.onset")
        coda = _require_phone_list(value["coda"], "segment.coda")
        for context, consonants in (("onset", onset), ("coda", coda)):
            for phone in consonants:
                base, _ = _split_phone(phone, f"segment.{context}")
                if base not in CONSONANTS:
                    raise ContractValidationError(f"segment.{context} may contain consonants only")

        nucleus = _require_nonempty_string(value["nucleus"], "segment.nucleus")
        if nucleus not in VOWELS:
            raise ContractValidationError("segment.nucleus must be an unstressed ARPAbet vowel")
        stress = _require_int(value["stress"], "segment.stress")
        if stress not in {0, 1, 2}:
            raise ContractValidationError("segment.stress must be 0, 1, or 2")

        expected_phones = onset + (f"{nucleus}{stress}",) + coda
        if phones != expected_phones:
            raise ContractValidationError(
                f"segment.phones {phones!r} does not match components {expected_phones!r}"
            )
        label = _require_nonempty_string(value["label"], "segment.label")
        if label != "-".join(phones):
            raise ContractValidationError("segment.label must be the hyphen-joined phone sequence")

        raw_token_id = value["token_id"]
        token_id = None if raw_token_id is None else _require_int(raw_token_id, "segment.token_id")
        confidence = _require_confidence(value["confidence"], "segment.confidence")
        return cls(
            target_index=target_index,
            start=start,
            end=end,
            phones=phones,
            label=label,
            onset=onset,
            nucleus=nucleus,
            coda=coda,
            stress=stress,
            token_id=token_id,
            confidence=confidence,
        )


@dataclass(frozen=True)
class AlignmentEvent:
    type: str
    target_index: int | None
    start: float | None
    end: float | None
    phones: tuple[str, ...]
    confidence: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AlignmentEvent":
        _require_exact_keys(value, _EVENT_KEYS, "alignment event")
        event_type = value["type"]
        if event_type not in _EVENT_TYPES:
            raise ContractValidationError(f"unknown alignment event type {event_type!r}")

        raw_target_index = value["target_index"]
        target_index = (
            None
            if raw_target_index is None
            else _require_int(raw_target_index, "alignment event.target_index")
        )
        phones = _require_phone_list(value["phones"], "alignment event.phones")
        confidence = _require_confidence(value["confidence"], "alignment event.confidence")

        if event_type == "insertion" and target_index is not None:
            raise ContractValidationError("insertion event.target_index must be null")
        if event_type != "insertion" and target_index is None:
            raise ContractValidationError(f"{event_type} event.target_index is required")

        raw_start, raw_end = value["start"], value["end"]
        if event_type == "deletion":
            if raw_start is not None or raw_end is not None or phones:
                raise ContractValidationError("deletion events must have null times and no observed phones")
            start = end = None
        else:
            start = _require_number(raw_start, "alignment event.start")
            end = _require_number(raw_end, "alignment event.end")
            if end <= start:
                raise ContractValidationError("alignment event.end must be greater than start")
        return cls(event_type, target_index, start, end, phones, confidence)


def validate_guided_result(payload: Mapping[str, Any]) -> None:
    """Validate JSON shape-dependent invariants that JSON Schema cannot express."""
    if not isinstance(payload, Mapping):
        raise ContractValidationError("result must be an object")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "result")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractValidationError(f"schema_version must be {SCHEMA_VERSION!r}")
    if payload["mode"] != "guided":
        raise ContractValidationError("mode must be 'guided'")
    _require_nonempty_string(payload["audio_id"], "audio_id")
    _require_nonempty_string(payload["target_text"], "target_text")

    raw_segments = _require_sequence(payload["segments"], "segments")
    segments: list[SyllableSegment] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise ContractValidationError(f"segments[{index}] must be an object")
        segments.append(SyllableSegment.from_mapping(raw_segment))

    previous_end = 0.0
    previous_target_index = -1
    for segment in segments:
        if segment.target_index <= previous_target_index:
            raise ContractValidationError("segments must have strictly increasing target_index values")
        if segment.start < previous_end:
            raise ContractValidationError("segments must be chronological and non-overlapping")
        previous_target_index = segment.target_index
        previous_end = segment.end

    raw_alignment = payload["alignment"]
    if not isinstance(raw_alignment, Mapping):
        raise ContractValidationError("alignment must be an object")
    _require_exact_keys(raw_alignment, _ALIGNMENT_KEYS, "alignment")
    status = raw_alignment["status"]
    if status not in {"success", "partial", "failed"}:
        raise ContractValidationError("alignment.status is invalid")
    target_count = _require_int(
        raw_alignment["target_syllable_count"], "alignment.target_syllable_count", minimum=1
    )
    aligned_count = _require_int(
        raw_alignment["aligned_syllable_count"], "alignment.aligned_syllable_count"
    )
    if aligned_count != len(segments):
        raise ContractValidationError("alignment.aligned_syllable_count must equal len(segments)")

    raw_events = _require_sequence(raw_alignment["events"], "alignment.events")
    events: list[AlignmentEvent] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise ContractValidationError(f"alignment.events[{index}] must be an object")
        event = AlignmentEvent.from_mapping(raw_event)
        if event.target_index is not None and event.target_index >= target_count:
            raise ContractValidationError("alignment event.target_index is outside the target sequence")
        events.append(event)

    segment_indices = {segment.target_index for segment in segments}
    deletion_targets = [event.target_index for event in events if event.type == "deletion"]
    deletion_indices = set(deletion_targets)
    if None in deletion_indices:
        raise AssertionError("deletion target index validation failed")
    if len(deletion_targets) != len(deletion_indices):
        raise ContractValidationError("a target syllable cannot have duplicate deletion events")
    if segment_indices & deletion_indices:
        raise ContractValidationError("a target syllable cannot be both aligned and deleted")
    if segment_indices | deletion_indices != set(range(target_count)):
        raise ContractValidationError("aligned and deleted target indices must cover the target sequence")
    for event in events:
        if event.type in {"repetition", "mismatch"} and event.target_index not in segment_indices:
            raise ContractValidationError(f"{event.type} must reference an aligned target syllable")

    if status == "success":
        if events or aligned_count != target_count:
            raise ContractValidationError("success requires every target aligned and no events")
    elif status == "partial":
        if not segments or (not events and aligned_count == target_count):
            raise ContractValidationError("partial requires an aligned segment and a recorded deviation")
    elif segments:
        raise ContractValidationError("failed alignment cannot contain aligned segments")
