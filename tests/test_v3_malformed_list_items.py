from __future__ import annotations

import unittest

from src.editing.planner import build_edit_plan
from src.editing.sound_design import build_audio_events
from src.editing.visual_grammar import build_visual_events
from src.hooks.v3 import filter_truthful_overlays, select_hook_v3
from src.originality.models import EditorialAngle
from src.v3.models import HookPlan, TimelineSegment


class V3MalformedListItemTests(unittest.TestCase):
    def test_hook_candidates_ignore_non_dict_items(self):
        clip = {"start": 10.0, "end": 20.0, "hook_score": 90, "hook_confidence": 0.9}
        proposal = {
            "hook_candidates": [
                "malformed hook candidate",
                123,
                None,
                {"mode": "EDITORIAL", "score": 70, "confidence": 0.8, "text": "Wait"},
            ]
        }
        result = select_hook_v3(
            clip=clip,
            proposal=proposal,
            transcript=[{"start": 10.0, "end": 11.0, "text": "Wait"}],
            context_start=9.0,
            context_end=21.0,
        )
        self.assertEqual(result.mode, "NATIVE")
        self.assertEqual(result.score, 90)

    def test_overlay_visual_and_audio_consumers_ignore_non_dict_items(self):
        overlays = filter_truthful_overlays(
            ["bad overlay", {"text": "Wait", "start": 0.1, "duration": 1.0}],
            "Wait",
        )
        self.assertEqual(len(overlays), 1)

        visuals = build_visual_events(
            {
                "visual_events": [
                    "bad visual",
                    {"event": "reaction", "time": 0.2, "intensity": 0.8, "edit": "bad nested edit"},
                ]
            }
        )
        self.assertEqual(len(visuals), 1)
        self.assertEqual(visuals[0].effect, "face_zoom")

        audio = build_audio_events(
            {
                "audio_events": [
                    "bad audio",
                    {"effect": "impact", "time": 0.3, "intensity": 0.5},
                ]
            },
            enabled=True,
        )
        self.assertEqual(len(audio), 1)
        self.assertEqual(audio[0].effect, "impact")

    def test_edit_plan_ignores_malformed_caption_emphasis(self):
        plan = build_edit_plan(
            clip_index=1,
            hook=HookPlan(mode="NATIVE", score=90, source_start=10.0, confidence=0.9),
            angle=EditorialAngle(primary_angle="moment", confidence=0.8),
            timeline=[TimelineSegment(10.0, 20.0, "context")],
            proposal={
                "caption_emphasis": [
                    "bad caption item",
                    None,
                    {"phrase": "RUN NOW", "emphasis": "strong"},
                ],
                "context_overlays": ["bad overlay"],
                "visual_events": ["bad visual"],
                "audio_events": ["bad audio"],
            },
            no_transformation_needed=False,
            content_profile="gaming",
            editorial_profile_name="default",
            enable_context_overlays=False,
            enable_semantic_effects=False,
            enable_sound_design=False,
            max_effects_per_event=2,
        )
        self.assertEqual(plan.caption_emphasis, [{"phrase": "RUN NOW", "emphasis": "strong"}])


if __name__ == "__main__":
    unittest.main()
