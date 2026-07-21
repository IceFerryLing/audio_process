"""HuBERT encoder with an explicit ARPAbet Phone CTC head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import HubertModel


@dataclass
class PhoneCTCOutput:
    """模型输出：帧级 logits、每条音频的有效帧数，以及可选 CTC loss。"""

    logits: torch.Tensor
    input_lengths: torch.Tensor
    loss: torch.Tensor | None = None


class HubertPhoneCTC(nn.Module):
    """本地 HuBERT encoder 加随机初始化的 ARPAbet Phone CTC 分类头。"""

    def __init__(self, encoder: HubertModel, vocabulary_size: int, *, blank_id: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.blank_id = blank_id
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, vocabulary_size)
        self.loss_function = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    @classmethod
    def from_local_pretrained(
        cls,
        model_path: Path,
        vocabulary_size: int,
        *,
        blank_id: int = 0,
        dropout: float = 0.1,
    ) -> "HubertPhoneCTC":
        encoder = HubertModel.from_pretrained(model_path, local_files_only=True)
        return cls(encoder, vocabulary_size, blank_id=blank_id, dropout=dropout)

    def freeze_encoder(self) -> None:
        """冻结 HuBERT，只训练随机初始化的任务头以验证数据和 loss。"""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_top_layers(self, count: int) -> None:
        """保持卷积前端冻结，只逐步解冻最顶部的 Transformer block。"""
        if count < 0 or count > len(self.encoder.encoder.layers):
            raise ValueError("unfreeze layer count is outside the HuBERT encoder")
        self.freeze_encoder()
        if count:
            for layer in self.encoder.encoder.layers[-count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        """按 HuBERT 卷积 kernel/stride 计算波形长度对应的有效输出帧数。"""
        lengths = sample_lengths
        # 每层一维卷积都会缩短时间轴；CTC 必须使用缩短后的帧长度。
        for kernel, stride in zip(self.encoder.config.conv_kernel, self.encoder.config.conv_stride):
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        return lengths

    def encode(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把波形 [B, S] 编码为隐藏状态 [B, T, H] 和有效帧数 [B]。"""
        encoder_trainable = any(parameter.requires_grad for parameter in self.encoder.parameters())
        if encoder_trainable:
            encoded = self.encoder(input_values=input_values, attention_mask=attention_mask)
        else:
            # encoder 完全冻结时关闭梯度，减少正确性过拟合实验的内存和计算开销。
            self.encoder.eval()
            with torch.no_grad():
                encoded = self.encoder(input_values=input_values, attention_mask=attention_mask)
        sample_lengths = attention_mask.to(torch.long).sum(dim=-1)
        # padding 后每条波形长度不同，不能把整块 padded hidden state 都交给 CTC。
        input_lengths = self.output_lengths(sample_lengths).clamp(max=encoded.last_hidden_state.shape[1])
        return encoded.last_hidden_state, input_lengths

    def classify(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> PhoneCTCOutput:
        """把隐藏状态投影为 [B, T, V] 音素 logits，并在有标签时计算 CTC loss。"""
        logits = self.classifier(self.dropout(hidden_states))
        loss = None
        if labels is not None:
            # collator 使用 -100 padding，因此非负位置才是真实目标音素 ID。
            label_mask = labels >= 0
            label_lengths = label_mask.sum(dim=-1)
            # CTC 至少需要足够的声学帧来消费完整目标序列。
            if torch.any(input_lengths < label_lengths):
                raise ValueError("CTC input length is shorter than its target")
            flattened_labels = labels.masked_select(label_mask)
            # nn.CTCLoss 要求时间维在最前：[T, B, V]，并使用 log probabilities。
            log_probabilities = logits.float().log_softmax(dim=-1).transpose(0, 1)
            loss = self.loss_function(
                log_probabilities,
                flattened_labels,
                input_lengths,
                label_lengths,
            )
        return PhoneCTCOutput(logits=logits, input_lengths=input_lengths, loss=loss)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> PhoneCTCOutput:
        """完整前向：波形编码后接 Phone CTC 分类与可选 loss。"""
        hidden_states, input_lengths = self.encode(input_values, attention_mask)
        return self.classify(hidden_states, input_lengths, labels)
