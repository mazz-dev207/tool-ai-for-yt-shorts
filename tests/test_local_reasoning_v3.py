from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai_usage import build_ai_usage
from src.hooks.analyzer import GeminiHookAnalyzer
from src.v3.editorial_reasoner import (
    GeminiEditorialReasoner,
    _validate_response_shape,
    build_editorial_prompt,
)


class _FakeHookLocal:
    def __init__(self):
        self.calls = 0

    def analyze(self, **_kwargs):
        self.calls += 1
        return {"candidates": [], "best_start": 10.0, "_source": "qwen_local"}, False


class LocalReasoningV3Tests(unittest.TestCase):
    def test_hook_start_compatibility_class_is_qwen_only(self):
        fake = _FakeHookLocal()
        with patch("src.hooks.analyzer.QwenHookAnalyzer", return_value=fake):
            analyzer = GeminiHookAnalyzer()
            raw, from_cache = analyzer.analyze(
                video_path=Path("missing.mp4"),
                transcript=[],
                original_start=10.0,
                highlight_end=20.0,
                candidate_starts=[10.0],
                profile="gaming",
                video_duration=30.0,
                clip_index=1,
            )
        self.assertEqual(analyzer.provider, "qwen_local")
        self.assertEqual(fake.calls, 1)
        self.assertFalse(from_cache)
        self.assertEqual(raw["_source"], "qwen_local")

    def test_v3_prompt_declares_local_same_request_satisfaction(self):
        prompt = build_editorial_prompt(
            clip={
                "start": 10.0,
                "end": 20.0,
                "hook_score": 80,
                "gemini": {"total_score": 88, "scores": {"payoff": 85}},
            },
            transcript_text="[10.0s-12.0s] Wait, what is that?",
            profile="gaming",
            context_start=5.0,
            context_end=25.0,
            retry_feedback=["low_context_independence"],
        )
        self.assertIn("LOCAL QWEN", prompt)
        self.assertIn("Final Multimodal Judge", prompt)
        self.assertIn("SAME QWEN REQUEST", prompt)
        self.assertIn("No invented facts", prompt)
        self.assertIn("semantic truth", prompt)
        self.assertIn("ABSOLUTE timestamps", prompt)
        self.assertIn("RESTRUCTURED SHORT", prompt)
        self.assertIn("playback_rate MUST be 1.0", prompt)
        self.assertIn("low_context_independence", prompt)

    def test_v3_qwen_proposal_gets_satisfaction_without_gemini(self):
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "editorial_angle": {
                            "primary_angle": "one mistake changes the fight",
                            "viewer_question": "Can he recover?",
                            "stakes": "The fight can turn here",
                            "payoff": "The exchange resolves",
                            "confidence": 0.82,
                        },
                        "originality_analysis": {
                            "why_interesting": "Immediate conflict with a nearby payoff",
                            "source_dependency": 45,
                            "context_independence": 82,
                            "current_opening_quality": 78,
                            "transformation_need": 55,
                            "recommended_transformations": ["compress setup"],
                            "no_transformation_needed": False,
                        },
                        "hook_candidates": [],
                        "timeline": [],
                        "context_overlays": [],
                        "visual_events": [],
                        "audio_events": [],
                        "caption_emphasis": [],
                        "recommended_transformations": ["compress setup"],
                    }
                )
            }
        }
        clip = {
            "start": 10.0,
            "end": 20.0,
            "hook_score": 82,
            "hook": "One mistake can turn this",
            "payoff": "The exchange resolves",
            "gemini": {
                "total_score": 90,
                "category": "conflict",
                "reason": "Clear fight escalation and payoff.",
                "has_complete_payoff": True,
                "requires_previous_context": False,
                "scores": {
                    "hook": 85,
                    "payoff": 90,
                    "emotion": 78,
                    "visual_action": 88,
                    "surprise": 70,
                    "standalone": 86,
                    "replayability": 72,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"x")
            with patch("src.v3.editorial_reasoner._LOCAL_CACHE_DIR", Path(tmp) / "cache"), patch(
                "src.v3.editorial_reasoner.ollama.chat", return_value=response
            ) as chat:
                proposal, from_cache = GeminiEditorialReasoner().analyze(
                    video_path=video,
                    transcript=[{"start": 10.0, "end": 20.0, "text": "Wait, keep going"}],
                    clip=clip,
                    content_profile="gaming",
                    video_duration=30.0,
                    clip_index=1,
                )
        self.assertFalse(from_cache)
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(proposal["_provider"], "qwen_local")
        self.assertEqual(proposal["_satisfaction_provider"], "qwen_local_same_request")
        self.assertIn("viewer_satisfaction", proposal)
        self.assertGreaterEqual(proposal["viewer_satisfaction"]["payoff_score"], 80)
        self.assertTrue(proposal["timeline"])
        self.assertEqual(proposal["timeline"][0]["playback_rate"], 1.0)

    def test_response_shape_guard_still_rejects_missing_contract(self):
        with self.assertRaises(ValueError):
            _validate_response_shape({"timeline": []})

    def test_ai_usage_reports_only_final_judge_gemini_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "video_gemini_metadata.json").write_text(
                json.dumps(
                    {
                        "gemini_api_attempted_count": 4,
                        "cache_hits": 2,
                        "failures": 0,
                        "quota_exhausted": False,
                    }
                ),
                encoding="utf-8",
            )
            (root / "video.json").write_text(
                json.dumps([{"v3": {}, "satisfaction": {}, "caption_hook": {}}] * 5),
                encoding="utf-8",
            )
            (root / "video_hook_metadata.json").write_text(
                json.dumps({"shorts": [{}, {}, {}, {}, {}]}), encoding="utf-8"
            )
            with patch("src.ai_usage.HIGHLIGHTS_DIR", root):
                usage = build_ai_usage("video")
        self.assertEqual(usage["gemini"]["api_requests"], 4)
        self.assertEqual(usage["gemini"]["calls_per_short"], 0.8)
        self.assertEqual(usage["local"]["gemini_hook_start_calls"], 0)
        self.assertEqual(usage["local"]["gemini_v3_editorial_calls"], 0)
        self.assertEqual(usage["local"]["gemini_satisfaction_calls"], 0)


if __name__ == "__main__":
    unittest.main()
