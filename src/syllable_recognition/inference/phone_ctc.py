"""Guided Phone CTC alignment directly from audio and target text."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import yaml
from transformers import AutoFeatureExtractor

from syllable_recognition.data.phone_sequences import PhoneVocabulary, sha256_file
from syllable_recognition.data.normalization import normalize_librispeech_text
from syllable_recognition.data.pronunciation import PronunciationResolver
from syllable_recognition.decoding.phone_ctc import (
    collapse_ctc_ids,
    forced_align_ctc,
    frame_interval_to_seconds,
)
from syllable_recognition.models.phone_ctc import HubertPhoneCTC


def align_phone_ctc(
    training_config_path: Path,
    checkpoint_path: Path,
    audio_path: Path,
    target_text: str,
) -> dict[str, Any]:
    config_path = training_config_path.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parents[2]
    resolve = lambda value: (root / value).resolve()
    model_path = resolve(raw["model"]["local_path"])
    vocabulary_path = resolve(raw["data"]["vocabulary"])
    vocabulary_payload = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary = PhoneVocabulary(
        tuple(vocabulary_payload["tokens"]),
        blank_id=int(vocabulary_payload["blank_id"]),
        unknown_id=int(vocabulary_payload["unknown_id"]),
        label_padding_id=int(vocabulary_payload["label_padding_id"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if checkpoint["config_sha256"] != expected_config_hash:
        raise ValueError("checkpoint and training config hashes differ")
    if checkpoint["vocabulary_sha256"] != sha256_file(vocabulary_path):
        raise ValueError("checkpoint and vocabulary hashes differ")

    model = HubertPhoneCTC.from_local_pretrained(
        model_path,
        len(vocabulary.tokens),
        blank_id=vocabulary.blank_id,
        dropout=float(raw["model"]["dropout"]),
    )
    missing, unexpected = model.load_state_dict(checkpoint["trainable_model_state"], strict=False)
    if unexpected or any(name.startswith("classifier.") for name in missing):
        raise ValueError("checkpoint does not contain a compatible Phone CTC head")
    model.freeze_encoder()
    model.eval()

    waveform, sample_rate = sf.read(audio_path, dtype="float32")
    if sample_rate != 16_000 or waveform.ndim != 1:
        raise ValueError("Phone CTC alignment requires mono 16 kHz audio")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    features = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        output = model(features.input_values, features.attention_mask.to(torch.long))
    frame_count = int(output.input_lengths[0])
    log_probabilities = output.logits[0, :frame_count].float().log_softmax(dim=-1)

    normalized = normalize_librispeech_text(target_text)
    pronunciations = PronunciationResolver.from_default_packages().resolve_many(normalized.words)
    target_phones = [phone for pronunciation in pronunciations for phone in pronunciation.phones]
    target_ids = vocabulary.encode(target_phones)
    if vocabulary.unknown_id in target_ids:
        raise ValueError("target text contains phones outside the train-only vocabulary")
    alignment = forced_align_ctc(log_probabilities, target_ids, blank_id=vocabulary.blank_id)
    frame_hop = math.prod(model.encoder.config.conv_stride)
    duration = len(waveform) / sample_rate
    segments: list[dict[str, Any]] = []
    for aligned, phone in zip(alignment, target_phones):
        start, end = frame_interval_to_seconds(
            aligned.start_frame,
            aligned.end_frame,
            sample_rate=sample_rate,
            frame_hop_samples=frame_hop,
            audio_duration=duration,
        )
        segments.append(
            {
                "target_index": aligned.target_index,
                "phone": phone,
                "start": start,
                "end": end,
                "confidence": round(aligned.confidence, 6),
                "start_frame": aligned.start_frame,
                "end_frame": aligned.end_frame,
            }
        )
    greedy_ids = collapse_ctc_ids(
        output.logits[0, :frame_count].argmax(dim=-1).tolist(),
        blank_id=vocabulary.blank_id,
    )
    return {
        "schema_version": "phone-ctc-alignment-v1",
        "audio_id": audio_path.stem,
        "mode": raw["mode"],
        "task": "phone_ctc_alignment",
        "target_text": normalized.normalized,
        "target_phones": target_phones,
        "greedy_phones": vocabulary.decode(greedy_ids),
        "timestamp_semantics": "ctc_emission_span",
        "frame_hop_seconds": frame_hop / sample_rate,
        "mfa_timestamps_used": False,
        "segments": segments,
    }
