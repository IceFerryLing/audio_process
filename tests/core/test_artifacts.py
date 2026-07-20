from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.core.artifacts import (  # noqa: E402
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)


class ArtifactTests(unittest.TestCase):
    def test_hash_helpers_match_hashlib(self) -> None:
        payload = b"syllable-recognition\n"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(payload)
            self.assertEqual(sha256_bytes(payload), expected)
            self.assertEqual(sha256_file(path), expected)

    def test_json_round_trip_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "artifact.json"
            write_json(path, {"z": 1, "a": ["AH0"]})
            self.assertEqual(read_json(path), {"a": ["AH0"], "z": 1})
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_jsonl_round_trip_ignores_blank_lines(self) -> None:
        rows = [{"id": "a", "phones": ["AH0"]}, {"id": "b", "phones": ["B"]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            write_jsonl(path, rows)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertEqual(read_jsonl(path), rows)


if __name__ == "__main__":
    unittest.main()
