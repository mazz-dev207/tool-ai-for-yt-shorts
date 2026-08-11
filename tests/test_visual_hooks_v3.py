import unittest

from src.editing.visual_hooks import (
    apply_visual_hook_timeline,
    build_visual_hook_event,
    select_visual_hook,
)
from src.v3.models import HookPlan, TimelineSegment


class VisualHookEngineTests(unittest.TestCase):
    def _timeline(self):
        return [
            TimelineSegment(10.0, 14.0, "context"),
            TimelineSegment(14.0, 19.0, "escalation"),
            TimelineSegment(19.0, 23.0, "payoff"),
            TimelineSegment(23.0, 25.0, "reaction"),
        ]

    def test_show_result_first_reuses_real_payoff_range(self):
        proposal = {
            "recommended_transformations": ["Show result first, then cut back to context"],
            "visual_events": [],
        }
        hook = HookPlan(mode="NATIVE", score=75, source_start=10.0, source_end=11.0)
        selected = select_visual_hook(
            proposal,
            hook=hook,
            timeline=self._timeline(),
            content_profile="entertainment",
        )
        self.assertEqual(selected["technique"], "show_result_first")
        self.assertTrue(selected["source_supported"])
        self.assertGreaterEqual(selected["source_start"], 19.0)
        updated, changed = apply_visual_hook_timeline(self._timeline(), selected)
        self.assertTrue(changed)
        self.assertEqual(updated[0].purpose, "cold_open")
        self.assertIn("show a real payoff", updated[0].semantic_note.lower())

    def test_camera_whip_becomes_one_opening_visual_event(self):
        proposal = {
            "recommended_transformations": ["Use a short camera whip at the opening"],
            "visual_events": [],
        }
        hook = HookPlan(mode="NATIVE", score=82)
        selected = select_visual_hook(
            proposal,
            hook=hook,
            timeline=self._timeline(),
            content_profile="gaming",
        )
        self.assertEqual(selected["technique"], "camera_whip")
        event = build_visual_hook_event(selected)
        self.assertIsNotNone(event)
        self.assertEqual(event.effect, "focus_crop")
        self.assertEqual(event.time, 0.0)
        self.assertEqual(event.metadata["visual_hook_technique"], "camera_whip")

    def test_immediate_surprise_can_trigger_camera_whip_but_not_podcast(self):
        proposal = {
            "recommended_transformations": [],
            "visual_events": [
                {
                    "time": 0.2,
                    "event": "surprise",
                    "intensity": 0.9,
                    "target": "auto",
                    "edit": {},
                }
            ],
        }
        hook = HookPlan(mode="NATIVE", score=80)
        gaming = select_visual_hook(
            proposal,
            hook=hook,
            timeline=self._timeline(),
            content_profile="gaming",
        )
        podcast = select_visual_hook(
            proposal,
            hook=hook,
            timeline=self._timeline(),
            content_profile="podcast",
        )
        self.assertEqual(gaming["technique"], "camera_whip")
        self.assertEqual(podcast["technique"], "none")

    def test_action_in_motion_requires_source_range_before_restructure(self):
        proposal = {
            "recommended_transformations": ["Start mid-action with action in motion"],
            "visual_events": [],
        }
        without_range = select_visual_hook(
            proposal,
            hook=HookPlan(mode="NATIVE", score=70),
            timeline=self._timeline(),
            content_profile="entertainment",
        )
        self.assertEqual(without_range["technique"], "action_in_motion")
        self.assertFalse(without_range["source_supported"])
        unchanged, changed = apply_visual_hook_timeline(self._timeline(), without_range)
        self.assertFalse(changed)
        self.assertEqual(len(unchanged), 4)

        with_range = select_visual_hook(
            proposal,
            hook=HookPlan(
                mode="RECONSTRUCTED",
                score=90,
                source_start=14.2,
                source_end=15.0,
            ),
            timeline=self._timeline(),
            content_profile="entertainment",
        )
        self.assertTrue(with_range["source_supported"])
        updated, changed = apply_visual_hook_timeline(self._timeline(), with_range)
        self.assertTrue(changed)
        self.assertEqual(updated[0].purpose, "cold_open")

    def test_object_interaction_and_unexpected_angle_are_source_grounded(self):
        hook = HookPlan(mode="NATIVE", score=80)
        for phrase, expected in [
            ("Use object interaction across the lens", "object_interaction"),
            ("Open on the unexpected angle", "unexpected_angle"),
        ]:
            selected = select_visual_hook(
                {"recommended_transformations": [phrase], "visual_events": []},
                hook=hook,
                timeline=self._timeline(),
                content_profile="entertainment",
            )
            self.assertEqual(selected["technique"], expected)
            self.assertFalse(selected["source_supported"])
            self.assertIsNone(build_visual_hook_event(selected))

    def test_explicit_visual_hook_is_normalized(self):
        selected = select_visual_hook(
            {
                "visual_hook": {
                    "technique": "camera_whip",
                    "confidence": 2.0,
                    "source_supported": True,
                    "apply_mode": "editorial_effect",
                    "direction": "right_to_left",
                    "reason": "explicit multimodal selection",
                }
            },
            hook=HookPlan(mode="NATIVE", score=90),
            timeline=self._timeline(),
            content_profile="gaming",
        )
        self.assertEqual(selected["technique"], "camera_whip")
        self.assertEqual(selected["confidence"], 1.0)
        self.assertFalse(selected["inferred"])


if __name__ == "__main__":
    unittest.main()
