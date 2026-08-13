import tempfile
import unittest
from pathlib import Path

from src.caption_hooks.models import CaptionHookResult
from src.caption_hooks.renderer import (
    append_caption_hook_to_ass,
    strip_caption_hook_from_ass,
)


class CaptionHookWebcamPolicyV3Tests(unittest.TestCase):
    def _base_ass(self) -> str:
        return (
            "[Script Info]\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n\n"
            "[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
            "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
            "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
            "MarginR,MarginV,Encoding\n"
            "Style: Default,Anton,92,&H00FFFFFF,&H0000FFFF,&H00000000,"
            "&H00000000,-1,0,0,0,100,100,0,0,1,4,1,5,120,120,0,1\n\n"
            "[Events]\n"
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,NORMAL SUBTITLE\n"
        )

    def test_top_card_has_background_and_top_position(self):
        result = CaptionHookResult(
            enabled=True,
            status="ok",
            text="ONE MISTAKE ENDS THE RUN",
            lines=["ONE MISTAKE", "ENDS THE RUN"],
            duration=1.4,
            position="top_safe",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.ass"
            path.write_text(self._base_ass(), encoding="utf-8")
            self.assertTrue(append_caption_hook_to_ass(path, result))
            content = path.read_text(encoding="utf-8")
            self.assertIn("Style: CaptionHookBox,", content)
            self.assertIn("Dialogue: 4,", content)
            self.assertIn("Dialogue: 5,", content)
            self.assertIn(r"\pos(540,240)", content)

    def test_webcam_copy_removes_only_caption_hook(self):
        result = CaptionHookResult(
            enabled=True,
            status="ok",
            text="FINAL ROUND",
            lines=["FINAL ROUND"],
            duration=1.4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "clip.ass"
            clean = Path(tmp) / "clip_no_hook.ass"
            source.write_text(self._base_ass(), encoding="utf-8")
            append_caption_hook_to_ass(source, result)
            strip_caption_hook_from_ass(source, clean)
            content = clean.read_text(encoding="utf-8")
            self.assertNotIn("CaptionHook", content)
            self.assertIn("NORMAL SUBTITLE", content)


if __name__ == "__main__":
    unittest.main()
