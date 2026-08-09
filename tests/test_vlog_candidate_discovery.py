import unittest

from src.highlight_selector import (
    build_window_context,
    discovery_profile_guidance,
    normalize_clip,
    normalize_content_type,
)


class VlogCandidateDiscoveryTests(unittest.TestCase):
    def test_vlog_alias_maps_to_entertainment(self):
        self.assertEqual(normalize_content_type("vlog"), "entertainment")
        self.assertEqual(normalize_content_type("creator vlog"), "entertainment")

    def test_entertainment_guidance_is_vlog_aware(self):
        guidance = discovery_profile_guidance("entertainment")
        self.assertIn("VLOG", guidance)
        self.assertIn("micro-stories", guidance)
        self.assertIn("conversational", guidance)

    def test_recovery_context_explicitly_favors_recall(self):
        window = {
            "start": 0.0,
            "end": 60.0,
            "segments": [
                {"start": 0.0, "end": 20.0, "text": "We changed the plan and found something unexpected."}
            ],
        }
        context = build_window_context(
            window,
            1,
            content_profile="entertainment",
            recovery=True,
        )
        self.assertIn("DISCOVERY PASS: RECOVERY", context)
        self.assertIn("ZERO-CANDIDATE RECOVERY MODE", context)
        self.assertIn("Gemini and Retention", context)

    def test_recovery_threshold_can_keep_moderate_vlog_candidate(self):
        window = {
            "start": 0.0,
            "end": 60.0,
            "segments": [],
        }
        raw = {
            "start": 10.0,
            "end": 32.0,
            "title": "Unexpected change of plans",
            "hook": "The plan suddenly changes",
            "payoff": "They discover what happens next",
            "reason": "Coherent vlog micro-story",
            "scores": {
                "hook": 58,
                "retention": 60,
                "entertainment": 66,
                "emotional_intensity": 45,
                "standalone_context": 62,
                "pacing": 55,
                "payoff": 61,
                "viral_potential": 45,
            },
        }
        primary = normalize_clip(
            raw,
            window,
            1,
            min_viral_score=50,
            content_profile="entertainment",
        )
        recovery = normalize_clip(
            raw,
            window,
            1,
            min_viral_score=40,
            discovery_mode="recovery",
            content_profile="entertainment",
        )
        self.assertIsNone(primary)
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery["discovery_mode"], "recovery")
        self.assertEqual(recovery["discovery_profile"], "entertainment")


if __name__ == "__main__":
    unittest.main()
