import unittest

from src.editing.planner import build_edit_plan
from src.hooks.v3 import editorial_text_truth_safe
from src.originality.models import EditorialAngle
from src.story.restructurer import restructure_story
from src.v3.models import HookPlan, TimelineSegment


class V3TruthAndBoundsTests(unittest.TestCase):
    def test_high_risk_fact_requires_transcript_support(self):
        self.assertFalse(
            editorial_text_truth_safe(
                "He only had one attempt left.",
                "He walks toward the door and laughs.",
            )
        )
        self.assertTrue(
            editorial_text_truth_safe(
                "He only had one attempt left.",
                "He says this is his only attempt left.",
            )
        )

    def test_context_overlay_is_filtered_by_truth_guard(self):
        proposal = {
            "_truth_context": "He says hello and walks away.",
            "context_overlays": [
                {
                    "text": "$10,000 was on the line.",
                    "start": 0.1,
                    "duration": 1.0,
                    "purpose": "stakes",
                }
            ],
        }
        plan = build_edit_plan(
            clip_index=1,
            hook=HookPlan(mode="NATIVE", score=80),
            angle=EditorialAngle(primary_angle="reaction"),
            timeline=[TimelineSegment(10, 15, "payoff")],
            proposal=proposal,
            no_transformation_needed=False,
            content_profile="gaming",
            editorial_profile_name="mazclips",
            enable_context_overlays=True,
            enable_semantic_effects=False,
            enable_sound_design=False,
            max_effects_per_event=2,
        )
        self.assertEqual(plan.context_overlays, [])

    def test_story_cannot_use_unseen_source_range(self):
        clip = {"start": 10.0, "end": 20.0}
        proposal = {
            "timeline": [
                {
                    "source_start": 50.0,
                    "source_end": 55.0,
                    "purpose": "payoff",
                    "playback_rate": 1.0,
                }
            ]
        }
        segments, debug = restructure_story(
            clip=clip,
            proposal=proposal,
            source_duration=100.0,
            allowed_start=5.0,
            allowed_end=30.0,
        )
        self.assertEqual(debug["fallback"], "outside_analyzed_context")
        self.assertEqual(segments[0].source_start, 10.0)

    def test_speed_change_is_ignored_to_preserve_sync(self):
        clip = {"start": 10.0, "end": 20.0}
        proposal = {
            "timeline": [
                {"source_start": 10.0, "source_end": 15.0, "purpose": "context", "playback_rate": 1.5},
                {"source_start": 15.0, "source_end": 20.0, "purpose": "payoff", "playback_rate": 1.0},
            ]
        }
        segments, debug = restructure_story(
            clip=clip,
            proposal=proposal,
            source_duration=30.0,
            allowed_start=5.0,
            allowed_end=25.0,
        )
        self.assertEqual(segments[0].playback_rate, 1.0)
        self.assertEqual(debug["ignored_speed_changes"], 1)


if __name__ == "__main__":
    unittest.main()
