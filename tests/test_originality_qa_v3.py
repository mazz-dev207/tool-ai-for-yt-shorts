import unittest

from src.editing.originality_qa import run_originality_qa
from src.v3.models import EditPlan, HookPlan, OriginalityResult, TimelineSegment


class OriginalityQAV3Tests(unittest.TestCase):
    def test_low_originality_requests_retry(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(mode="NATIVE", score=65),
            timeline=[TimelineSegment(1.0, 5.0, "payoff")],
        )
        originality = OriginalityResult(
            originality_score=45,
            context_independence=55,
        )
        report = run_originality_qa(
            plan=plan,
            originality=originality,
            min_score=60,
            source_duration=20.0,
            max_effects_per_event=2,
        )
        self.assertFalse(report.passed)
        self.assertIn("originality_below_threshold", report.problems)

    def test_strong_standalone_no_transform_can_pass(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(mode="NATIVE", score=90),
            timeline=[TimelineSegment(1.0, 5.0, "payoff")],
            no_transformation_needed=True,
        )
        originality = OriginalityResult(
            originality_score=48,
            context_independence=82,
        )
        report = run_originality_qa(
            plan=plan,
            originality=originality,
            min_score=60,
            source_duration=20.0,
            max_effects_per_event=2,
        )
        self.assertTrue(report.passed)
        self.assertTrue(report.checks.get("intentional_simple_edit"))


if __name__ == "__main__":
    unittest.main()
