"""Correctness-stage HuBERT Phone CTC training and checkpointing."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from transformers import AutoFeatureExtractor

from syllable_recognition.data.ctc_dataset import PhoneCTCCollator, PhoneCTCDataset
from syllable_recognition.data.phone_sequences import sha256_file
from syllable_recognition.decoding.phone_ctc import collapse_ctc_ids
from syllable_recognition.metrics.per import phone_error_rate
from syllable_recognition.models.phone_ctc import HubertPhoneCTC


class PhoneCTCTrainingError(RuntimeError):
    """Raised when a correctness-stage Phone CTC run violates its contract."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable
    }


def _save_checkpoint(
    path: Path,
    *,
    model: HubertPhoneCTC,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    config_hash: str,
    manifest_hash: str,
    vocabulary_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_type": "hubert_phone_ctc",
            "trainable_model_state": _trainable_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
            "vocabulary_sha256": vocabulary_hash,
            "torch_random_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
        },
        path,
    )


def _verify_checkpoint_restore(path: Path, model: HubertPhoneCTC) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    current = model.state_dict()
    for name, expected in checkpoint["trainable_model_state"].items():
        if name not in current or not torch.equal(current[name].cpu(), expected):
            return False
    return True


def _decode_batch(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    dataset: PhoneCTCDataset,
) -> list[list[str]]:
    frame_ids = logits.argmax(dim=-1).detach().cpu()
    hypotheses: list[list[str]] = []
    for ids, length in zip(frame_ids, input_lengths.detach().cpu()):
        collapsed = collapse_ctc_ids(
            ids[: int(length)].tolist(),
            blank_id=dataset.vocabulary.blank_id,
        )
        hypotheses.append(dataset.vocabulary.decode(collapsed))
    return hypotheses


def _cache_batches(
    loader: DataLoader,
    model: HubertPhoneCTC,
    device: torch.device,
) -> list[dict[str, Any]]:
    cached: list[dict[str, Any]] = []
    model.encoder.eval()
    for batch in loader:
        hidden_states, input_lengths = model.encode(
            batch["input_values"].to(device),
            batch["attention_mask"].to(device),
        )
        cached.append(
            {
                "hidden_states": hidden_states.detach(),
                "input_lengths": input_lengths.detach(),
                "labels": batch["labels"].to(device),
                "reference_phones": batch["reference_phones"],
            }
        )
    return cached


def _evaluate_final_state(
    batches: list[dict[str, Any]],
    model: HubertPhoneCTC,
    dataset: PhoneCTCDataset,
) -> tuple[float, float]:
    losses: list[float] = []
    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = model.classify(
                batch["hidden_states"],
                batch["input_lengths"],
                batch["labels"],
            )
            if output.loss is None:
                raise PhoneCTCTrainingError("evaluation CTC loss is missing")
            losses.append(float(output.loss.detach().cpu()))
            references.extend(batch["reference_phones"])
            hypotheses.extend(_decode_batch(output.logits, output.input_lengths, dataset))
    return sum(losses) / len(losses), phone_error_rate(references, hypotheses)


def train_phone_ctc(config_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Run a bounded correctness experiment; this is not formal model selection."""
    resolved_config = config_path.resolve()
    raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    if raw.get("stage") != "4-phone-ctc-correctness" or raw.get("task") != "phone_ctc":
        raise PhoneCTCTrainingError("training config must select Phone CTC correctness stage")
    if raw.get("mode") not in {"guided", "open"}:
        raise PhoneCTCTrainingError("training config must explicitly select guided or open mode")
    root = resolved_config.parents[2]
    resolve = lambda value: (root / value).resolve()
    manifest_path = resolve(raw["data"]["manifest"])
    vocabulary_path = resolve(raw["data"]["vocabulary"])
    model_path = resolve(raw["model"]["local_path"])
    run_directory = resolve(raw["output"]["run_directory"])
    report_path = run_directory / "metrics.json"
    config_hash = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
    manifest_hash = sha256_file(manifest_path)
    vocabulary_hash = sha256_file(vocabulary_path)
    weight_path = model_path / "pytorch_model.bin"
    if sha256_file(weight_path) != raw["model"]["weight_sha256"]:
        raise PhoneCTCTrainingError("local HuBERT weight hash differs from the training config")
    if report_path.is_file() and not overwrite:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("config_sha256") == config_hash
            and existing.get("manifest_sha256") == manifest_hash
            and existing.get("vocabulary_sha256") == vocabulary_hash
            and existing.get("status") in {"passed_smoke", "passed_overfit"}
        ):
            return {"status": "skipped", "run_directory": str(run_directory.relative_to(root))}
        raise FileExistsError("existing training run does not match; use --overwrite explicitly")

    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    _set_seed(int(raw["seed"]))
    device_name = raw["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise PhoneCTCTrainingError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    dataset = PhoneCTCDataset(manifest_path, vocabulary_path, root=root)
    if raw["correctness_gate"]["require_zero_mfa_supervision"] and any(
        row.get("mfa_timestamps_used") is not False for row in dataset.rows
    ):
        raise PhoneCTCTrainingError("training manifest contains MFA timestamp supervision")
    if bool(raw["training"]["mixed_precision"]):
        raise PhoneCTCTrainingError("mixed precision is not enabled in the correctness-stage trainer")
    if raw["training"].get("scheduler", "constant") != "constant":
        raise PhoneCTCTrainingError("correctness-stage trainer currently supports only a constant scheduler")
    maximum_items = int(raw["data"].get("max_items", len(dataset)))
    if maximum_items < 1 or maximum_items > len(dataset):
        raise PhoneCTCTrainingError("data.max_items is outside the manifest")
    training_data = Subset(dataset, range(maximum_items))
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    collator = PhoneCTCCollator(
        feature_extractor,
        label_padding_id=dataset.vocabulary.label_padding_id,
    )
    generator = torch.Generator().manual_seed(int(raw["seed"]))
    loader = DataLoader(
        training_data,
        batch_size=int(raw["training"]["batch_size"]),
        shuffle=bool(raw["training"]["shuffle"]),
        num_workers=int(raw["training"]["num_workers"]),
        collate_fn=collator,
        generator=generator,
    )
    model = HubertPhoneCTC.from_local_pretrained(
        model_path,
        len(dataset.vocabulary.tokens),
        blank_id=dataset.vocabulary.blank_id,
        dropout=float(raw["model"]["dropout"]),
    )
    if raw["model"]["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable()
    if raw["model"]["freeze_encoder"]:
        model.freeze_encoder()
    else:
        unfreeze_count = int(raw["model"]["unfreeze_top_layers"])
        if unfreeze_count < 1:
            raise PhoneCTCTrainingError("an unfrozen encoder requires at least one top layer")
        model.unfreeze_top_layers(unfreeze_count)
    model.to(device)

    head_parameters = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    parameter_groups: list[dict[str, Any]] = [
        {"params": head_parameters, "lr": float(raw["training"]["head_learning_rate"])}
    ]
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": float(raw["training"]["encoder_learning_rate"])}
        )
    optimizer = AdamW(parameter_groups, weight_decay=float(raw["training"]["weight_decay"]))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    cache_features = bool(raw["training"].get("cache_frozen_features", False))
    if cache_features and encoder_parameters:
        raise PhoneCTCTrainingError("feature caching is only valid with a frozen encoder")
    cached_batches = _cache_batches(loader, model, device) if cache_features else []

    global_step = 0
    losses: list[float] = []
    model.train()
    maximum_steps = int(raw["training"]["max_steps"])
    last_epoch = 0
    for epoch in range(int(raw["training"]["epochs"])):
        last_epoch = epoch
        epoch_batches = cached_batches if cache_features else loader
        for batch in epoch_batches:
            optimizer.zero_grad(set_to_none=True)
            if cache_features:
                output = model.classify(
                    batch["hidden_states"],
                    batch["input_lengths"],
                    batch["labels"],
                )
            else:
                output = model(
                    batch["input_values"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                )
            if output.loss is None or not torch.isfinite(output.loss):
                raise PhoneCTCTrainingError("Phone CTC loss is missing or non-finite")
            output.loss.backward()
            gradient_norm = clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(raw["training"]["gradient_clip_norm"]),
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(output.loss.detach().cpu()))
            global_step += 1
            if global_step >= maximum_steps:
                break
        if global_step >= maximum_steps:
            break

    if not losses:
        raise PhoneCTCTrainingError("training produced no optimizer steps")
    if cache_features:
        final_loss, final_per = _evaluate_final_state(cached_batches, model, dataset)
    else:
        evaluation_batches = _cache_batches(loader, model, device)
        final_loss, final_per = _evaluate_final_state(evaluation_batches, model, dataset)

    overfit_required = bool(raw["correctness_gate"]["overfit_required"])
    overfit_passed = False
    if overfit_required:
        overfit_passed = (
            final_per <= float(raw["correctness_gate"]["maximum_train_per"])
            and final_loss <= float(raw["correctness_gate"]["maximum_last_loss"])
        )

    last_checkpoint = run_directory / "last" / "checkpoint.pt"
    best_checkpoint = run_directory / "best" / "checkpoint.pt"
    checkpoint_arguments = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": last_epoch,
        "global_step": global_step,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "vocabulary_hash": vocabulary_hash,
    }
    _save_checkpoint(last_checkpoint, **checkpoint_arguments)
    _save_checkpoint(best_checkpoint, **checkpoint_arguments)
    restore_passed = _verify_checkpoint_restore(last_checkpoint, model)
    if raw["correctness_gate"]["require_checkpoint_restore"] and not restore_passed:
        raise PhoneCTCTrainingError("checkpoint restore verification failed")

    status = "passed_overfit" if overfit_passed else "passed_smoke"
    if overfit_required and not overfit_passed:
        status = "failed_overfit"
    report = {
        "schema_version": 1,
        "status": status,
        "stage": raw["stage"],
        "task": raw["task"],
        "mode": raw["mode"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": raw["model"]["model_id"],
        "model_revision": raw["model"]["revision"],
        "mfa_timestamps_used": False,
        "split": raw["data"]["split"],
        "global_steps": global_step,
        "mean_train_loss": sum(losses) / len(losses),
        "last_train_loss": losses[-1],
        "final_train_loss": final_loss,
        "final_train_per": final_per,
        "gradient_norm_last_step": float(gradient_norm),
        "checkpoint_restore_passed": restore_passed,
        "overfit_gate_passed": overfit_passed,
        "overfit_gate_note": raw["correctness_gate"]["note"],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_items": maximum_items,
        "cached_frozen_features": cache_features,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "vocabulary_sha256": vocabulary_hash,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
    }
    _write_json(report_path, report)
    if overfit_required and not overfit_passed:
        raise PhoneCTCTrainingError(
            f"overfit gate failed: final PER={final_per:.4f}, final loss={final_loss:.4f}"
        )
    return report
