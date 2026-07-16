from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import jsonschema
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.contracts import (  # noqa: E402
    ContractValidationError,
    validate_guided_result,
)


class GuidedContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(
            (ROOT / "schemas/guided_syllable_result_v1.schema.json").read_text(encoding="utf-8")
        )
        cls.valid = json.loads(
            (ROOT / "tests/fixtures/contracts/guided_valid.json").read_text(encoding="utf-8")
        )

    def test_schema_is_valid_draft_2020_12(self) -> None:
        jsonschema.Draft202012Validator.check_schema(self.schema)

    def test_reference_payload_passes_schema_and_semantics(self) -> None:
        jsonschema.validate(self.valid, self.schema)
        validate_guided_result(self.valid)

    def test_overlapping_segments_are_rejected(self) -> None:
        payload = json.loads(
            (ROOT / "tests/fixtures/contracts/guided_invalid_overlap.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.validate(payload, self.schema)
        with self.assertRaisesRegex(ContractValidationError, "non-overlapping"):
            validate_guided_result(payload)

    def test_component_mismatch_is_rejected(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["segments"][0]["nucleus"] = "AE"
        with self.assertRaisesRegex(ContractValidationError, "does not match components"):
            validate_guided_result(payload)

    def test_out_of_range_confidence_is_rejected(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["segments"][0]["confidence"] = 1.01
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(payload, self.schema)
        with self.assertRaisesRegex(ContractValidationError, "must be <= 1"):
            validate_guided_result(payload)

    def test_partial_alignment_requires_explicit_deletion(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["segments"] = payload["segments"][:2]
        payload["alignment"] = {
            "status": "partial",
            "target_syllable_count": 3,
            "aligned_syllable_count": 2,
            "events": [
                {
                    "type": "deletion",
                    "target_index": 2,
                    "start": None,
                    "end": None,
                    "phones": [],
                    "confidence": 0.82
                }
            ]
        }
        jsonschema.validate(payload, self.schema)
        validate_guided_result(payload)

    def test_success_cannot_hide_alignment_events(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["alignment"]["events"] = [
            {
                "type": "mismatch",
                "target_index": 1,
                "start": 0.29,
                "end": 0.51,
                "phones": ["L", "AO1"],
                "confidence": 0.65
            }
        ]
        with self.assertRaisesRegex(ContractValidationError, "success requires"):
            validate_guided_result(payload)

    def test_failed_alignment_cannot_contain_segments(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["alignment"]["status"] = "failed"
        with self.assertRaisesRegex(ContractValidationError, "failed alignment"):
            validate_guided_result(payload)

    def test_duplicate_deletion_event_is_rejected(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["segments"] = payload["segments"][:2]
        deletion = {
            "type": "deletion",
            "target_index": 2,
            "start": None,
            "end": None,
            "phones": [],
            "confidence": 0.82
        }
        payload["alignment"] = {
            "status": "partial",
            "target_syllable_count": 3,
            "aligned_syllable_count": 2,
            "events": [deletion, copy.deepcopy(deletion)]
        }
        with self.assertRaisesRegex(ContractValidationError, "duplicate deletion"):
            validate_guided_result(payload)

    def test_contract_config_fixes_guided_mode_and_stage_gate(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/contracts/guided_mvp_v1.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["mode"], "guided")
        self.assertEqual(config["input"]["audio"], {"sample_rate": 16000, "channels": 1, "time_unit": "seconds"})
        self.assertTrue(config["stage_gate"]["stop_on_failure"])
        self.assertFalse(config["stage_gate"]["downstream_commands_allowed"])

    def test_syllable_definition_separates_stress(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/data/syllabifier_arpabet_v1.yaml").read_text(encoding="utf-8")
        )
        self.assertFalse(config["cross_word_resyllabification"])
        self.assertTrue(config["stress"]["stored_separately_from_nucleus"])
        self.assertEqual(config["empty_classes"]["onset"], "<empty>")
        self.assertNotIn(["NG"], config["legal_intervocalic_onsets"])

    def test_evaluation_protocol_protects_test_clean(self) -> None:
        protocol = yaml.safe_load(
            (ROOT / "configs/evaluation/guided_mvp_v1.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(protocol["splits"]["train"]["librispeech"], "train-clean-100")
        self.assertEqual(protocol["splits"]["validation"]["librispeech"], "dev-clean")
        self.assertEqual(protocol["splits"]["test"]["librispeech"], "test-clean")
        self.assertTrue(protocol["splits"]["test_for_selection_forbidden"])
        self.assertEqual(protocol["metrics"]["boundary"]["matching_tolerances_ms"], [20, 50])


if __name__ == "__main__":
    unittest.main()
