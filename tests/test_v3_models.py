import unittest

from src.v3.models import EditPlan, HookPlan, TimelineSegment, VisualEvent


class V3ModelTests(unittest.TestCase):
    def test_valid_plan(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(mode="EDITORIAL", score=88, text="One attempt left.", confidence=0.9),
            timeline=[TimelineSegment(10.0, 12.0, "context"), TimelineSegment(15.0, 18.0, "payoff")],
            visual_events=[VisualEvent(time=2.0, effect="punch_in", intensity=0.8)],
        )
        self.assertEqual(plan.validate(source_duration=30.0), [])
        self.assertAlmostEqual(plan.duration, 5.0)

    def test_effect_budget_detects_overedit(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(),
            timeline=[TimelineSegment(1.0, 5.0, "payoff")],
            visual_events=[VisualEvent(1.0, "punch_in"), VisualEvent(1.05, "face_zoom"), VisualEvent(1.08, "focus_crop")],
        )
        self.assertIn("effect_budget_exceeded", plan.validate(source_duration=20.0, max_effects_per_event=2))

    def test_duplicate_non_replay_is_invalid(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(),
            timeline=[TimelineSegment(1, 3, "context"), TimelineSegment(1, 3, "context")],
        )
        self.assertIn("duplicate_content", plan.validate(source_duration=20))


if __name__ == "__main__":
    unittest.main()
