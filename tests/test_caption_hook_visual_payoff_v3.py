import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.caption_hooks.generator import generate_caption_hook
from src.caption_hooks.models import CaptionHookCandidate


class CaptionHookVisualPayoffV3Tests(unittest.TestCase):
    def test_gaming_visual_payoff_can_generate_hook_without_dialogue(self):
        clip = {
            "start": 20.0,
            "end": 27.0,
            "title": "Final Attempt",
            "hook": "",
            "payoff": "The player survives the final attempt",
            "satisfaction": {
                "status": "ok",
                "payoff_score": 94,
                "expectation_match_score": 90,
                "context_independence_score": 88,
                "hook_promise": "One mistake ends the run",
                "actual_payoff": "The player survives the final attempt",
                "risks": {"missing_context": False, "clickbait_gap": False},
                "payoff": {
                    "reason": "The visual gameplay result resolves the final attempt."
                },
            },
            "v3": {"hook_score_v3": 70},
        }
        edit_plan = {
            "hook": {"mode": "NATIVE", "score": 70, "text": ""},
            "timeline": [
                {
                    "source_start": 20.0,
                    "source_end": 27.0,
                    "purpose": "payoff",
                }
            ],
            "viewer_question": "Can the player survive the final attempt?",
            "stakes": "One mistake ends the run",
            "payoff": "The player survives the final attempt",
            "metadata": {"content_profile": "gaming"},
        }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.caption_hooks.generator.V3_CAPTION_HOOK_CACHE_DIR", Path(tmp)
        ), patch(
            "src.caption_hooks.generator._generate_local_candidates",
            return_value=[
                CaptionHookCandidate(
                    text="ONE MISTAKE ENDS THE RUN",
                    type="stakes",
                    mode="COMPLEMENT",
                    reason="Truthful stakes caption for a visual payoff.",
                )
            ],
        ):
            result = generate_caption_hook(
                video_name="visual_gameplay",
                clip_index=1,
                clip=clip,
                transcript=[],
                edit_plan=edit_plan,
                content_profile="gaming",
            )

        self.assertTrue(result.enabled)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.type, "stakes")
        self.assertEqual(result.text, "ONE MISTAKE ENDS THE RUN")
        self.assertGreaterEqual(result.score, 75)


if __name__ == "__main__":
    unittest.main()
