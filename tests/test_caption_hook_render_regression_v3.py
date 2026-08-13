from __future__ import annotations

import unittest

from src.caption_hooks.models import CaptionHookResult
from src.caption_hooks.renderer import _background_line, _styled_text, _text_line, _times


class CaptionHookRenderRegressionTests(unittest.TestCase):
    def _result(self) -> CaptionHookResult:
        return CaptionHookResult(
            enabled=True,
            status="ok",
            mode="COMPLEMENT",
            type="stakes",
            text="MANA DRAIN AVOIDED, FIREBALL CAST",
            lines=["MANA DRAIN AVOIDED,", "FIREBALL CAST"],
            emphasis=["FIREBALL"],
            start=0.05,
            # A stale cached value must not shorten the canonical render.
            duration=1.4,
            position="top_safe",
            score=90,
        )

    def test_caption_hook_render_duration_is_3_3_seconds(self):
        start, end = _times(self._result())
        self.assertAlmostEqual(end - start, 3.3, places=6)

    def test_text_override_block_closes_once_before_visible_text(self):
        line = _text_line(self._result())
        expected = r"\t(130,260,\fscx100\fscy100)}MANA DRAIN AVOIDED,"
        broken = r"\t(130,260,\fscx100\fscy100)}}MANA DRAIN AVOIDED,"
        self.assertIn(expected, line)
        self.assertNotIn(broken, line)

    def test_background_override_block_closes_once_before_vector_shape(self):
        line = _background_line(self._result())
        self.assertIn(r"\bord0\shad0\fad(80,180)}m ", line)
        self.assertNotIn(r"\bord0\shad0\fad(80,180)}}m ", line)

    def test_model_braces_do_not_leak_into_visible_caption_text(self):
        result = CaptionHookResult(
            enabled=True,
            text="}MANA DRAIN AVOIDED, FIREBALL CAST{",
            lines=["}MANA DRAIN AVOIDED,", "FIREBALL CAST{"],
            emphasis=[],
        )
        visible = _styled_text(result)
        self.assertNotIn("{", visible)
        self.assertNotIn("}", visible)
        self.assertIn("MANA DRAIN AVOIDED", visible)
        self.assertIn("FIREBALL CAST", visible)


if __name__ == "__main__":
    unittest.main()
