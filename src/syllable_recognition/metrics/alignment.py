"""Paired phone-boundary metrics for CTC versus an external reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def paired_phone_boundary_metrics(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    *,
    tolerances_ms: Sequence[int] = (20, 50),
) -> dict[str, Any]:
    """在音素标签完全一致时，一一比较 start/end 边界误差。

    这是正确性阶段的 paired 指标。它尚不负责解决插入、删除、重复或标签错位；
    正式 Guided 评价应先完成事件匹配，再对可配对边界计算误差。
    """
    if len(reference) != len(hypothesis) or not reference:
        raise ValueError("paired phone boundaries require equal non-empty sequences")
    reference_phones = [str(item.get("phone")) for item in reference]
    hypothesis_phones = [str(item.get("phone")) for item in hypothesis]
    # 当前指标按序号配对，因此标签不一致时必须停止，避免比较错位边界。
    if reference_phones != hypothesis_phones:
        raise ValueError("paired phone labels differ")
    # 每个音素贡献 start 和 end 两个边界误差，统一转换为毫秒。
    errors_ms = [
        abs(float(reference_item[key]) - float(hypothesis_item[key])) * 1000
        for reference_item, hypothesis_item in zip(reference, hypothesis)
        for key in ("start", "end")
    ]
    tolerance_metrics: dict[str, Any] = {}
    for tolerance in tolerances_ms:
        matched = sum(error <= tolerance for error in errors_ms)
        score = matched / len(errors_ms)
        # paired 场景中预测和参考边界数量完全相同，所以 P、R、F1 数值相同。
        tolerance_metrics[str(tolerance)] = {
            "matched_boundaries": matched,
            "total_boundaries": len(errors_ms),
            "precision": score,
            "recall": score,
            "f1": score,
        }
    return {
        "phone_count": len(reference),
        "boundary_count": len(errors_ms),
        "boundary_mae_ms": sum(errors_ms) / len(errors_ms),
        "maximum_boundary_error_ms": max(errors_ms),
        "tolerances_ms": tolerance_metrics,
    }
