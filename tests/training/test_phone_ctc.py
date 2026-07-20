from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.training.phone_ctc import (  # noqa: E402
    _save_checkpoint,
    _verify_checkpoint_restore,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 3)
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


class PhoneCTCTrainingTests(unittest.TestCase):
    def test_checkpoint_contains_and_restores_only_trainable_state(self) -> None:
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _save_checkpoint(
                path,
                model=model,  # type: ignore[arg-type]
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                global_step=1,
                config_hash="config",
                manifest_hash="manifest",
                vocabulary_hash="vocabulary",
            )
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(set(checkpoint["trainable_model_state"]), {"head.weight", "head.bias"})
            self.assertIn("scheduler_state", checkpoint)
            self.assertIn("python_random_state", checkpoint)
            self.assertTrue(_verify_checkpoint_restore(path, model))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
