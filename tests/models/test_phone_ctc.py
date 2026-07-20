from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.models.phone_ctc import HubertPhoneCTC  # noqa: E402


class FakeConfig:
    hidden_size = 4
    conv_kernel = [2, 2]
    conv_stride = [2, 2]


class FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeConfig()


class PhoneCTCModelTests(unittest.TestCase):
    def test_output_lengths_follow_encoder_convolutions(self) -> None:
        model = HubertPhoneCTC(FakeEncoder(), 5, blank_id=0, dropout=0.0)
        lengths = model.output_lengths(torch.tensor([16, 20]))
        self.assertEqual(lengths.tolist(), [4, 5])


if __name__ == "__main__":
    unittest.main()
