import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.v3.execution import has_executable_v3_changes


class V3ExecutionGateTests(unittest.TestCase):
    def _write(self, root: Path, clips: list[dict]) -> Path:
        highlights = root / "highlights"
        highlights.mkdir(exist_ok=True)
        (highlights / "video.json").write_text(
            json.dumps(clips), encoding="utf-8"
        )
        return highlights

    def test_all_native_no_transform_uses_v2_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = self._write(
                root,
                [
                    {
                        "v3": {
                            "hook_mode": "NATIVE",
                            "no_transformation_needed": True,
                        }
                    }
                ],
            )
            with patch("src.v3.execution.HIGHLIGHTS_DIR", highlights):
                self.assertFalse(has_executable_v3_changes("video"))

    def test_reconstructed_hook_requires_v3_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = self._write(
                root,
                [
                    {
                        "v3": {
                            "hook_mode": "RECONSTRUCTED",
                            "no_transformation_needed": True,
                        }
                    }
                ],
            )
            with patch("src.v3.execution.HIGHLIGHTS_DIR", highlights):
                self.assertTrue(has_executable_v3_changes("video"))

    def test_story_transform_requires_v3_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = self._write(
                root,
                [
                    {
                        "v3": {
                            "hook_mode": "NATIVE",
                            "no_transformation_needed": False,
                        }
                    }
                ],
            )
            with patch("src.v3.execution.HIGHLIGHTS_DIR", highlights):
                self.assertTrue(has_executable_v3_changes("video"))


if __name__ == "__main__":
    unittest.main()
