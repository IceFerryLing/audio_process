"""Phone error rate using token-level Levenshtein distance."""

from __future__ import annotations

from collections.abc import Sequence


def phone_edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """用动态规划计算音素级 Levenshtein 编辑距离。"""
    # previous[j] 表示已处理参考前缀到 hypothesis[:j] 的最小编辑次数。
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_phone in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_phone in enumerate(hypothesis, start=1):
            current.append(
                min(
                    # 删除当前参考音素、插入当前预测音素、或匹配/替换。
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
    """计算语料级 PER：总编辑次数除以参考音素总数。"""
    if len(references) != len(hypotheses):
        raise ValueError("reference and hypothesis batches differ in length")
    reference_count = sum(len(reference) for reference in references)
    if reference_count == 0:
        raise ValueError("PER requires at least one reference phone")
    # 先跨 utterance 累加编辑次数，再除以总参考长度，而不是平均每句 PER。
    edits = sum(
        phone_edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    )
    return edits / reference_count
