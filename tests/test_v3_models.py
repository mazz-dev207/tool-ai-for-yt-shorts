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

    def test_partial_overlap_between_normal_segments_is_invalid(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(),
            timeline=[
                TimelineSegment(4.0, 8.0, "context"),
                TimelineSegment(7.5, 10.0, "escalation"),
                TimelineSegment(11.0, 13.0, "payoff"),
            ],
        )
        self.assertIn(
            "overlapping_source_segments",
            plan.validate(source_duration=20.0),
        )

    def test_cold_open_preview_may_overlap_payoff(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(),
            timeline=[
                TimelineSegment(18.0, 19.0, "cold_open"),
                TimelineSegment(10.0, 14.0, "context"),
                TimelineSegment(18.0, 20.0, "payoff"),
            ],
        )
        self.assertNotIn(
            "overlapping_source_segments",
            plan.validate(source_duration=30.0),
        )


if __name__ == "__main__":
    unittest.main()