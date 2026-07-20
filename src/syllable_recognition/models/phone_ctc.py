"""HuBERT encoder with an explicit ARPAbet Phone CTC head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import HubertModel


@dataclass
class PhoneCTCOutput:
    logits: torch.Tensor
    input_lengths: torch.Tensor
    loss: torch.Tensor | None = None


class HubertPhoneCTC(nn.Module):
    """A local HuBERT encoder and randomly initialized phone classifier."""

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
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_top_layers(self, count: int) -> None:
        if count < 0 or count > len(self.encoder.encoder.layers):
            raise ValueError("unfreeze layer count is outside the HuBERT encoder")
        self.freeze_encoder()
        if count:
            for layer in self.encoder.encoder.layers[-count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        lengths = sample_lengths
        for kernel, stride in zip(self.encoder.config.conv_kernel, self.encoder.config.conv_stride):
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        return lengths

    def encode(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_trainable = any(parameter.requires_grad for parameter in self.encoder.parameters())
        if encoder_trainable:
            encoded = self.encoder(input_values=input_values, attention_mask=attention_mask)
        else:
            self.encoder.eval()
            with torch.no_grad():
                encoded = self.encoder(input_values=input_values, attention_mask=attention_mask)
        sample_lengths = attention_mask.to(torch.long).sum(dim=-1)
        input_lengths = self.output_lengths(sample_lengths).clamp(max=encoded.last_hidden_state.shape[1])
        return encoded.last_hidden_state, input_lengths

    def classify(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> PhoneCTCOutput:
        logits = self.classifier(self.dropout(hidden_states))
        loss = None
        if labels is not None:
            label_mask = labels >= 0
            label_lengths = label_mask.sum(dim=-1)
            if torch.any(input_lengths < label_lengths):
                raise ValueError("CTC input length is shorter than its target")
            flattened_labels = labels.masked_select(label_mask)
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
        hidden_states, input_lengths = self.encode(input_values, attention_mask)
        return self.classify(hidden_states, input_lengths, labels)
