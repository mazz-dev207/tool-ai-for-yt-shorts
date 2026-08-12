import sys
import unittest
from unittest.mock import patch

from main import parse_args
from src.editing.post_processor import (
    apply_cli_visual_hook_override,
    normalize_visual_hook_override,
)


class VisualHookCliV3Tests(unittest.TestCase):
    def _plan(self):
        return {
            "hook": {"score": 40},
            "metadata": {"content_profile": "general"},
            "visual_events": [
                {
                    "time": 0.20,
                    "effect": "punch_in",
                    "intensity": 0.4,
                    "duration": 0.3,
                    "target": "center",
                    "metadata": {"semantic_event": "old_opening"},
                },
                {
                    "time": 2.0,
                    "effect": "focus_crop",
                    "intensity": 0.5,
                    "duration": 0.4,
                    "target": "center",
                    "metadata": {"semantic_event": "later_payoff"},
                },
            ],
            "no_transformation_needed": True,
        }

    def test_cli_parser_accepts_visual_hook_alias(self):
        with patch.object(
            sys,
            "argv",
            ["main.py", "video2", "--visual-hook", "camera-whip"],
        ):
            args = parse_args()
        self.assertEqual(args.visual_hook, "camera-whip")

        with patch.object(
            sys,
            "argv",
            ["main.py", "video2", "--visual-hook-effect", "punch-in"],
        ):
            args = parse_args()
        self.assertEqual(args.visual_hook, "punch-in")

    def test_camera_whip_is_forced_even_for_low_score_non_gaming_plan(self):
        plan = self._plan()
        changed = apply_cli_visual_hook_override(plan, 1, "camera-whip")
        self.assertTrue(changed)
        first = plan["visual_events"][0]
        self.assertEqual(first["time"], 0.0)
        self.assertEqual(first["effect"], "focus_crop")
        self.assertEqual(first["metadata"]["visual_hook_technique"], "camera_whip")
        self.assertTrue(first["metadata"]["cli_forced"])
        self.assertFalse(plan["no_transformation_needed"])

    def test_forced_override_replaces_opening_but_preserves_later_events(self):
        plan = self._plan()
        apply_cli_visual_hook_override(plan, 2, "focus-crop")
        self.assertEqual(len(plan["visual_events"]), 2)
        self.assertEqual(plan["visual_events"][0]["effect"], "focus_crop")
        self.assertEqual(plan["visual_events"][1]["time"], 2.0)
        self.assertEqual(
            plan["visual_events"][1]["metadata"]["semantic_event"],
            "later_payoff",
        )

    def test_each_forceable_effect_normalizes_from_cli_spelling(self):
        self.assertEqual(normalize_visual_hook_override("camera-whip"), "camera_whip")
        self.assertEqual(normalize_visual_hook_override("punch-in"), "punch_in")
        self.assertEqual(normalize_visual_hook_override("focus-crop"), "focus_crop")
        self.assertEqual(normalize_visual_hook_override("face-zoom"), "face_zoom")

    def test_punch_and_face_zoom_are_real_opening_events(self):
        for requested, expected in [
            ("punch-in", "punch_in"),
            ("face-zoom", "face_zoom"),
        ]:
            plan = self._plan()
            apply_cli_visual_hook_override(plan, 3, requested)
            event = plan["visual_events"][0]
            self.assertEqual(event["effect"], expected)
            self.assertEqual(event["time"], 0.0)
            self.assertEqual(event["metadata"]["visual_hook_technique"], expected)

    def test_auto_does_not_modify_existing_plan(self):
        plan = self._plan()
        before = [dict(item) for item in plan["visual_events"]]
        changed = apply_cli_visual_hook_override(plan, 1, "auto")
        self.assertFalse(changed)
        self.assertEqual(plan["visual_events"], before)


if __name__ == "__main__":
    unittest.main()
