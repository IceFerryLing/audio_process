"""Small-sample Wav2Vec2 CTC fine-tuning on LibriSpeech.

The default dummy dataset is a tiny LibriSpeech excerpt intended only to verify
that downloading, preprocessing, training, and evaluation all work end to end.
Use --dataset full after the smoke test succeeds.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCTC, AutoProcessor


DATASETS = {
    "dummy": ("hf-internal-testing/librispeech_asr_dummy", "clean", "validation"),
    "full": ("openslr/librispeech_asr", "clean", "train.100"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--dataset", choices=DATASETS, default="dummy")
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-audio-seconds", type=float, default=4.0)
    parser.add_argument("--output-dir", default="outputs/wav2vec2-librispeech-smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-feature-encoder", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_audio(audio: dict) -> tuple[np.ndarray, int]:
    """Decode datasets.Audio(decode=False) without requiring torchcodec."""
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    samples, sampling_rate = sf.read(source, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples, sampling_rate


def prepare_dataset(args: argparse.Namespace, processor, model_config) -> tuple[Dataset, Dataset]:
    dataset_name, config, split = DATASETS[args.dataset]
    total = args.train_samples + args.eval_samples
    candidate_count = total if args.max_audio_seconds <= 0 else total * 10
    dataset = load_dataset(dataset_name, config, split=f"{split}[:{candidate_count}]")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    def model_output_length(input_length: int) -> int:
        length = input_length
        for kernel, stride in zip(model_config.conv_kernel, model_config.conv_stride):
            length = (length - kernel) // stride + 1
        return length

    def prepare(example: dict) -> dict:
        speech, sampling_rate = load_audio(example["audio"])
        if sampling_rate != processor.feature_extractor.sampling_rate:
            raise ValueError(
                f"Expected {processor.feature_extractor.sampling_rate} Hz audio, got {sampling_rate} Hz"
            )
        inputs = processor(speech, sampling_rate=sampling_rate)
        labels = processor(text=example["text"].upper()).input_ids
        repeated_tokens = sum(left == right for left, right in zip(labels, labels[1:]))
        return {
            "input_values": inputs.input_values[0],
            "labels": labels,
            "duration": len(speech) / sampling_rate,
            "ctc_frames": model_output_length(len(speech)),
            "ctc_required": len(labels) + repeated_tokens,
        }

    dataset = dataset.map(prepare, remove_columns=dataset.column_names)
    dataset = dataset.filter(
        lambda example: (
            (args.max_audio_seconds <= 0 or example["duration"] <= args.max_audio_seconds)
            and example["ctc_frames"] >= example["ctc_required"]
        )
    )
    if len(dataset) < total:
        raise ValueError(
            f"Only {len(dataset)} utterances match max_audio_seconds={args.max_audio_seconds}; "
            "increase --max-audio-seconds or request fewer samples"
        )
    dataset = dataset.remove_columns(["duration", "ctc_frames", "ctc_required"])
    train = dataset.select(range(args.train_samples))
    evaluation = dataset.select(range(args.train_samples, total))
    return train, evaluation


def make_collator(processor):
    def collate(features: list[dict]) -> dict[str, torch.Tensor]:
        inputs = [torch.tensor(item["input_values"], dtype=torch.float32) for item in features]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in features]
        input_values = pad_sequence(inputs, batch_first=True)
        attention_mask = pad_sequence(
            [torch.ones_like(item, dtype=torch.long) for item in inputs], batch_first=True
        )
        padded_labels = pad_sequence(
            labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id
        )
        label_mask = pad_sequence(
            [torch.ones_like(item, dtype=torch.bool) for item in labels], batch_first=True
        )
        padded_labels[~label_mask] = -100
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

    return collate


@torch.no_grad()
def evaluate(model, loader, processor, device: torch.device) -> dict:
    model.eval()
    batch = next(iter(loader))
    batch = {key: value.to(device) for key, value in batch.items()}
    output = model(**batch)
    predicted_ids = output.logits.argmax(dim=-1)
    labels = batch["labels"].clone()
    labels[labels == -100] = processor.tokenizer.pad_token_id
    return {
        "eval_loss": round(output.loss.item(), 4),
        "prediction": processor.batch_decode(predicted_ids)[0],
        "reference": processor.batch_decode(labels, group_tokens=False)[0],
    }


def main() -> None:
    args = parse_args()
    if min(args.train_samples, args.eval_samples, args.max_steps, args.batch_size) < 1:
        raise ValueError("Sample counts, max steps, and batch size must all be positive")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, model={args.model_name}, dataset={args.dataset}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCTC.from_pretrained(args.model_name).to(device)
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    train_set, eval_set = prepare_dataset(args, processor, model.config)
    collator = make_collator(processor)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    model.train()
    step = 0
    while step < args.max_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss; check audio/label alignment")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"step={step}/{args.max_steps}, train_loss={loss.item():.4f}")
            if step >= args.max_steps:
                break

    metrics = evaluate(model, eval_loader, processor, device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "run_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
