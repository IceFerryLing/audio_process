from __future__ import annotations

import sys
import unittest
from pathlib import Path

from click.testing import CliRunner


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syllable_recognition.cli import main  # noqa: E402


class CliStructureTests(unittest.TestCase):
    def test_top_level_command_groups_remain_available(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        for command in ("align", "data", "evaluate", "train"):
            self.assertIn(command, result.output)

    def test_domain_subcommands_remain_available(self) -> None:
        expected = {
            "align": ("phone-ctc",),
            "data": (
                "build",
                "build-phone-sequences",
                "mfa-download-spec",
                "mfa-spec",
                "prepare-mfa",
                "record-review",
                "validate",
            ),
            "evaluate": ("phone-ctc-mfa",),
            "train": ("phone-ctc",),
        }
        runner = CliRunner()
        for group, commands in expected.items():
            with self.subTest(group=group):
                result = runner.invoke(main, [group, "--help"])
                self.assertEqual(result.exit_code, 0, result.output)
                for command in commands:
                    self.assertIn(command, result.output)


if __name__ == "__main__":
    unittest.main()
