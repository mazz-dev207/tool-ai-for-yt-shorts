import unittest

from src.originality.scoring import calculate_originality_score, score_edit_plan
from src.originality.models import OriginalityAnalysis
from src.v3.models import EditPlan, HookPlan, TimelineSegment, ContextOverlay, VisualEvent


class OriginalityScoringTests(unittest.TestCase):
    def test_weighted_score_range(self):
        score = calculate_originality_score(
            editorial_transformation=80, narrative_restructuring=80, context_independence=80,
            visual_transformation=70, hook_originality=90, audio_transformation=40,
            brand_consistency=80, source_dependency=20,
        )
        self.assertGreaterEqual(score, 70)
        self.assertLessEqual(score, 100)

    def test_editorial_plan_scores_higher_than_plain_extraction(self):
        analysis = OriginalityAnalysis(source_dependency=35, context_independence=80)
        plain = EditPlan(
            clip_index=1, hook=HookPlan(mode="NATIVE", score=70),
            timeline=[TimelineSegment(10, 20, "context")],
            metadata={"editorial_profile": "mazclips", "content_profile": "gaming"},
        )
        transformed = EditPlan(
            clip_index=1, hook=HookPlan(mode="EDITORIAL", score=90),
            timeline=[
                TimelineSegment(18, 19, "cold_open"), TimelineSegment(10, 13, "context"),
                TimelineSegment(14, 17, "escalation"), TimelineSegment(18, 20, "payoff"),
            ],
            context_overlays=[ContextOverlay("One attempt left.", 0.1, 1.2, "stakes")],
            visual_events=[VisualEvent(5.0, "punch_in", 0.8)],
            editorial_angle="close call",
            metadata={"editorial_profile": "mazclips", "content_profile": "gaming"},
        )
        self.assertGreater(score_edit_plan(transformed, analysis).originality_score, score_edit_plan(plain, analysis).originality_score)


if __name__ == "__main__":
    unittest.main()
