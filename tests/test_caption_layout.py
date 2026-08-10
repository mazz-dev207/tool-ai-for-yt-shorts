import tempfile
import unittest
from pathlib import Path

from src.caption.animations import Animation
from src.caption.ass_writer import ASSWriter
from src.caption.grouping import Word, WordGroup
from src.caption.styles import DEFAULT_STYLE


class CaptionLayoutTests(unittest.TestCase):
    def test_default_style_is_centered_with_safe_horizontal_margins(self):
        self.assertEqual(DEFAULT_STYLE.alignment, 5)
        self.assertGreaterEqual(DEFAULT_STYLE.margin_l, 100)
        self.assertGreaterEqual(DEFAULT_STYLE.margin_r, 100)

    def test_ass_writer_enables_smart_wrapping(self):
        group = WordGroup(
            words=[
                Word("THIS", 0.0, 0.2),
                Word("IS", 0.2, 0.35),
                Word("CENTERED", 0.35, 0.7),
            ],
            start=0.0,
            end=0.7,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "captions.ass"
            ASSWriter(DEFAULT_STYLE, Animation.CAPCUT).write([group], output)
            content = output.read_text(encoding="utf-8")

        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)
        self.assertIn("WrapStyle: 0", content)
        self.assertIn("{\\q0}", content)
        self.assertIn(",5,120,120,0,", content)


if __name__ == "__main__":
    unittest.main()
