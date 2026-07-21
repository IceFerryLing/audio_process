from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class PhoneCTCOneHourConfigTests(unittest.TestCase):
    def test_data_and_training_configs_share_one_hour_artifacts(self) -> None:
        data_config = yaml.safe_load(
            (ROOT / "configs/data/librispeech_phone_ctc_1h.yaml").read_text(encoding="utf-8")
        )
        train_config = yaml.safe_load(
            (ROOT / "configs/train/hubert_phone_ctc_1h_frozen.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(data_config["stage"], "3-phone-sequence-manifest")
        self.assertEqual(train_config["stage"], "4-phone-ctc-correctness")
        self.assertEqual(data_config["input"]["split"], "train-clean-100")
        self.assertEqual(train_config["data"]["split"], "train-clean-100")
        self.assertEqual(
            data_config["output"]["manifest"],
            train_config["data"]["manifest"],
        )
        self.assertEqual(
            data_config["output"]["vocabulary"],
            train_config["data"]["vocabulary"],
        )
        self.assertEqual(data_config["data_gate"]["expected_items"], 289)
        self.assertEqual(train_config["data"]["max_items"], 289)

    def test_frozen_run_does_not_claim_formal_model_selection(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/train/hubert_phone_ctc_1h_frozen.yaml").read_text(encoding="utf-8")
        )

        self.assertTrue(config["model"]["freeze_encoder"])
        self.assertEqual(config["model"]["unfreeze_top_layers"], 0)
        self.assertTrue(config["training"]["cache_frozen_features"])
        self.assertFalse(config["correctness_gate"]["overfit_required"])
        self.assertIn("no dev selection", config["correctness_gate"]["note"])
        self.assertNotEqual(
            config["output"]["run_directory"],
            "runs/hubert-phone-ctc-12-overfit",
        )


if __name__ == "__main__":
    unittest.main()
