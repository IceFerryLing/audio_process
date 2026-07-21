"""Greedy Phone CTC decoding without language-model assumptions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def collapse_ctc_ids(
    frame_ids: Sequence[int],
    *,
    blank_id: int,
    ignored_ids: set[int] | None = None,
) -> list[int]:
    """执行标准 CTC 贪心折叠：先合并连续重复，再移除 blank/忽略项。"""
    ignored = set(ignored_ids or ()) | {blank_id}
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in frame_ids:
        # 相同 token 被 blank 隔开时会重新出现；因此 previous 必须记录 blank。
        if token_id != previous and token_id not in ignored:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


@dataclass(frozen=True)
class AlignedCTCToken:
    """一个目标 token 在最优 CTC 路径上的发射帧区间。"""

    target_index: int
    token_id: int
    start_frame: int
    end_frame: int
    confidence: float


def forced_align_ctc(
    log_probabilities: torch.Tensor,
    target_ids: Sequence[int],
    *,
    blank_id: int,
) -> list[AlignedCTCToken]:
    """用 Viterbi 将一个已知目标序列约束到帧级 CTC 对数概率上。

    输入为 ``[frames, vocabulary]``；输出是每个目标 token 的发射帧区间。
    这些区间是 CTC emission span，不等同于完整的声学音素边界。
    """
    if log_probabilities.ndim != 2:
        raise ValueError("CTC alignment expects [frames, vocabulary] log probabilities")
    if not target_ids:
        return []
    if any(token_id == blank_id for token_id in target_ids):
        raise ValueError("the CTC target cannot contain blank")
    frame_count, vocabulary_size = log_probabilities.shape
    if any(token_id < 0 or token_id >= vocabulary_size for token_id in target_ids):
        raise ValueError("the CTC target contains an out-of-vocabulary ID")

    # 把目标 [A, B] 扩展为 [blank, A, blank, B, blank]，便于表达停留和重复。
    extended: list[int] = [blank_id]
    for token_id in target_ids:
        extended.extend((token_id, blank_id))
    state_count = len(extended)
    scores = torch.full(
        (frame_count, state_count),
        -torch.inf,
        dtype=log_probabilities.dtype,
        device=log_probabilities.device,
    )
    backpointers = torch.full(
        (frame_count, state_count),
        -1,
        dtype=torch.long,
        device=log_probabilities.device,
    )
    scores[0, 0] = log_probabilities[0, blank_id]
    if state_count > 1:
        scores[0, 1] = log_probabilities[0, extended[1]]

    for frame in range(1, frame_count):
        for state, symbol in enumerate(extended):
            # 合法转移：停留当前状态，前进一格，或在满足条件时跳过中间 blank。
            candidates = [(scores[frame - 1, state], state)]
            if state >= 1:
                candidates.append((scores[frame - 1, state - 1], state - 1))
            # 相邻相同 token 不能直接跨两格，否则两个重复音素会被 CTC 合并。
            if state >= 2 and symbol != blank_id and symbol != extended[state - 2]:
                candidates.append((scores[frame - 1, state - 2], state - 2))
            best_score, previous_state = max(candidates, key=lambda item: float(item[0]))
            scores[frame, state] = best_score + log_probabilities[frame, symbol]
            backpointers[frame, state] = previous_state

    # 合法路径可以结束在最后一个 blank，也可以结束在最后一个目标 token。
    ending_states = [state_count - 1, state_count - 2]
    final_state = max(ending_states, key=lambda state: float(scores[-1, state]))
    if not torch.isfinite(scores[-1, final_state]):
        raise ValueError("no valid CTC path consumes the complete target sequence")
    path = [final_state]
    # 从最终状态沿 backpointer 回溯，恢复每一帧选择的扩展目标状态。
    for frame in range(frame_count - 1, 0, -1):
        previous_state = int(backpointers[frame, path[-1]])
        if previous_state < 0:
            raise ValueError("CTC backtrace terminated before the first frame")
        path.append(previous_state)
    path.reverse()

    aligned: list[AlignedCTCToken] = []
    for target_index, token_id in enumerate(target_ids):
        target_state = 2 * target_index + 1
        # 扩展序列中奇数状态对应真实目标 token，偶数状态对应 blank。
        frames = [frame for frame, state in enumerate(path) if state == target_state]
        if not frames:
            raise ValueError(f"target index {target_index} received no CTC emission frame")
        # 当前 confidence 是发射帧上的平均 token 概率，不是发音正确率。
        confidence = float(log_probabilities[frames, token_id].exp().mean().detach().cpu())
        aligned.append(
            AlignedCTCToken(
                target_index=target_index,
                token_id=token_id,
                start_frame=frames[0],
                end_frame=frames[-1] + 1,
                confidence=confidence,
            )
        )
    return aligned


def frame_interval_to_seconds(
    start_frame: int,
    end_frame: int,
    *,
    sample_rate: int,
    frame_hop_samples: int,
    audio_duration: float,
) -> tuple[float, float]:
    """根据 encoder 总 stride 将半开帧区间转换成秒，并裁剪到音频时长。"""
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("frame interval must be positive and ordered")
    start = start_frame * frame_hop_samples / sample_rate
    end = end_frame * frame_hop_samples / sample_rate
    return round(min(start, audio_duration), 6), round(min(end, audio_duration), 6)
