"""Dataset and dynamic collator for waveform-to-phone CTC training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch.utils.data import Dataset

from syllable_recognition.core.artifacts import read_json, read_jsonl

from .phone_sequences import PhoneVocabulary


class PhoneCTCDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: Path, vocabulary_path: Path, *, root: Path) -> None:
        self.rows = read_jsonl(manifest_path)
        vocabulary_payload = read_json(vocabulary_path)
        self.vocabulary = PhoneVocabulary(
            tuple(vocabulary_payload["tokens"]),
            blank_id=int(vocabulary_payload["blank_id"]),
            unknown_id=int(vocabulary_payload["unknown_id"]),
            label_padding_id=int(vocabulary_payload["label_padding_id"]),
        )
        self.root = root.resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = sf.read(self.root / row["audio"], dtype="float32")
        if sample_rate != 16_000 or waveform.ndim != 1:
            raise ValueError(f"{row['id']} must be mono 16 kHz")
        return {
            "id": row["id"],
            "input_values": waveform,
            "labels": self.vocabulary.encode(row["phones"]),
            "phones": row["phones"],
        }


class PhoneCTCCollator:
    def __init__(self, feature_extractor: Any, *, label_padding_id: int = -100) -> None:
        self.feature_extractor = feature_extractor
        self.label_padding_id = label_padding_id

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("cannot collate an empty batch")
        features = self.feature_extractor(
            [item["input_values"] for item in items],
            sampling_rate=16_000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        maximum_labels = max(len(item["labels"]) for item in items)
        labels = torch.full(
            (len(items), maximum_labels),
            self.label_padding_id,
            dtype=torch.long,
        )
        for index, item in enumerate(items):
            labels[index, : len(item["labels"])] = torch.tensor(item["labels"], dtype=torch.long)
        return {
            "ids": [item["id"] for item in items],
            "input_values": features.input_values,
            "attention_mask": features.attention_mask.to(torch.long),
            "labels": labels,
            "reference_phones": [item["phones"] for item in items],
        }
