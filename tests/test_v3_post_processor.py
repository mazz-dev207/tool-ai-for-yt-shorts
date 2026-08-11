import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.editing.post_processor import (
    _dialogue_duck_expression,
    _write_visual_commands,
)


class V3PostProcessorTests(unittest.TestCase):
    def test_punch_in_uses_runtime_crop_commands(self):
        plan = {
            "visual_events": [
                {
                    "time": 1.0,
                    "effect": "punch_in",
                    "intensity": 0.8,
                    "duration": 0.5,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.editing.post_processor.TEMP_DIR", Path(tmp)):
                path = _write_visual_commands(plan, 1)
                self.assertIsNotNone(path)
                content = path.read_text(encoding="utf-8")
                self.assertIn("crop@v3 w", content)
                self.assertIn("crop@v3 h", content)
                self.assertIn("crop@v3 x", content)
                self.assertIn("1.500 crop@v3 w 1080", content)

    def test_unsupported_effect_is_not_executed(self):
        plan = {
            "visual_events": [
                {"time": 1.0, "effect": "freeze_frame", "intensity": 1.0, "duration": 0.5}
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.editing.post_processor.TEMP_DIR", Path(tmp)):
                self.assertIsNone(_write_visual_commands(plan, 1))

    def test_dialogue_ducking_only_covers_sfx_windows(self):
        expression = _dialogue_duck_expression([(1.0, 1.4), (3.0, 3.3)])
        self.assertIn("between(t,1.000,1.400)", expression)
        self.assertIn("between(t,3.000,3.300)", expression)
        self.assertIn("0.820", expression)
        self.assertTrue(expression.endswith("1))"))


if __name__ == "__main__":
    unittest.main()