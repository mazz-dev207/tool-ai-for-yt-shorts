import unittest

from src.editing.post_processor import _ensure_runtime_visual_hook
from src.editing.visual_hooks import build_visual_hook_event, select_visual_hook
from src.v3.models import HookPlan, TimelineSegment


class VisualHookFallbackV3Tests(unittest.TestCase):
    def _timeline(self):
        return [
            TimelineSegment(10.0, 16.0, "escalation"),
            TimelineSegment(16.0, 20.0, "payoff"),
        ]

    def test_strong_gaming_hook_gets_deterministic_camera_whip_without_gemini(self):
        hook = HookPlan(mode="NATIVE", score=97, source_start=10.0)
        selected = select_visual_hook(
            {},
            hook=hook,
            timeline=self._timeline(),
            content_profile="gaming",
        )
        self.assertEqual(selected["technique"], "camera_whip")
        self.assertTrue(selected["fallback"])
        event = build_visual_hook_event(selected)
        self.assertIsNotNone(event)
        self.assertEqual(event.metadata["visual_hook_technique"], "camera_whip")

    def test_moderate_gaming_hook_does_not_get_decorative_fallback(self):
        selected = select_visual_hook(
            {},
            hook=HookPlan(mode="NATIVE", score=94, source_start=10.0),
            timeline=self._timeline(),
            content_profile="gaming",
        )
        self.assertEqual(selected["technique"], "none")

    def test_non_gaming_hook_never_gets_score_only_camera_whip(self):
        selected = select_visual_hook(
            {},
            hook=HookPlan(mode="NATIVE", score=99, source_start=10.0),
            timeline=self._timeline(),
            content_profile="entertainment",
        )
        self.assertEqual(selected["technique"], "none")

    def test_runtime_fallback_repairs_quota_plan(self):
        plan = {
            "hook": {"mode": "NATIVE", "score": 96, "source_start": 10.0},
            "visual_events": [],
            "no_transformation_needed": True,
            "metadata": {
                "content_profile": "gaming",
                "fallback_reason": "gemini_daily_quota_exhausted",
            },
        }
        changed = _ensure_runtime_visual_hook(plan, clip_index=2)
        self.assertTrue(changed)
        self.assertEqual(len(plan["visual_events"]), 1)
        event = plan["visual_events"][0]
        self.assertEqual(event["metadata"]["visual_hook_technique"], "camera_whip")
        self.assertTrue(plan["metadata"]["visual_hook_runtime_fallback"])
        self.assertFalse(plan["no_transformation_needed"])

    def test_runtime_fallback_respects_explicit_normal_path_decision(self):
        plan = {
            "hook": {"mode": "NATIVE", "score": 99},
            "visual_events": [],
            "metadata": {
                "content_profile": "gaming",
                "visual_hook": {"technique": "none", "reason": "not useful"},
            },
        }
        changed = _ensure_runtime_visual_hook(plan, clip_index=1)
        self.assertFalse(changed)
        self.assertEqual(plan["visual_events"], [])


if __name__ == "__main__":
    unittest.main()
