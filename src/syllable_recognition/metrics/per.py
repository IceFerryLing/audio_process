"""Phone error rate using token-level Levenshtein distance."""

from __future__ import annotations

from collections.abc import Sequence


def phone_edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_phone in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_phone in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + (reference_phone != hypothesis_phone),
                )
            )
        previous = current
    return previous[-1]


def phone_error_rate(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("reference and hypothesis batches differ in length")
    reference_count = sum(len(reference) for reference in references)
    if reference_count == 0:
        raise ValueError("PER requires at least one reference phone")
    edits = sum(
        phone_edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    )
    return edits / reference_count
