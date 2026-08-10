import unittest
from unittest.mock import patch

from src.hooks.smartcut_bridge import refine_edit_plan_preserving_hook


class HookSmartCutBridgeTests(unittest.TestCase):
    def test_protected_hook_start_survives_smartcut(self):
        smartcut_result = [{"start": 19.2, "end": 25.0}]
        with patch(
            "src.hooks.smartcut_bridge.refine_edit_plan",
            return_value=smartcut_result,
        ):
            result = refine_edit_plan_preserving_hook(
                video_name="video",
                clip_index=1,
                original_segments=[{"start": 20.0, "end": 25.0}],
                transcript=[],
                video_duration=100.0,
                optimized_start=20.0,
            )
        self.assertEqual(result[0]["start"], 20.0)
        self.assertEqual(result[0]["end"], 25.0)

    def test_invalid_protected_start_keeps_safe_smartcut_result(self):
        smartcut_result = [{"start": 19.2, "end": 25.0}]
        with patch(
            "src.hooks.smartcut_bridge.refine_edit_plan",
            return_value=smartcut_result,
        ):
            result = refine_edit_plan_preserving_hook(
                video_name="video",
                clip_index=1,
                original_segments=[{"start": 20.0, "end": 25.0}],
                transcript=[],
                video_duration=100.0,
                optimized_start=24.5,
            )
        self.assertEqual(result, smartcut_result)


if __name__ == "__main__":
    unittest.main()
