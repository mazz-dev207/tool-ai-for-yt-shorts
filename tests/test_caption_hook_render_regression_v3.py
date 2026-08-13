import unittest

from src.caption_hooks.models import CaptionHookResult
from src.caption_hooks.renderer import _styled_text, _times


class CaptionHookRenderRegressionTests(unittest.TestCase):
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

    def test_caption_hook_render_duration_is_3_3_seconds(self):
        result = CaptionHookResult(
            enabled=True,
            text="MANA DRAIN AVOIDED",
            lines=["MANA DRAIN AVOIDED"],
            start=0.05,
            duration=1.1,  # stale cache value must not shorten the rendered hook
        )
        start, end = _times(result)
        self.assertAlmostEqual(end - start, 3.3, places=3)


if __name__ == "__main__":
    unittest.main()
