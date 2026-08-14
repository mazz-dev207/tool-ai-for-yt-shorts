from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.hooks.analyzer import GeminiHookAnalyzer
from src.hooks.qwen_analyzer import QwenHookAnalyzer


class _FakeLocalAnalyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, **_kwargs):
        self.calls += 1
        return {
            "candidates": [
                {
                    "start": 20.5,
                    "core_scores": {
                        "immediate_action": 17,
                        "curiosity_gap": 18,
                        "emotional_reaction": 10,
                        "conflict_tension": 13,
                        "visual_surprise": 2,
                        "context_independence": 9,
                        "payoff_proximity": 9,
                    },
                    "human_modifiers": {"naturalness": 8},
                    "penalties": {},
                    "hook_type": "STAKES",
                    "hook_confidence": 0.82,
                }
            ],
            "best_start": 20.5,
            "_source": "qwen_local_fallback",
        }, False


class HookQwenFallbackV3Tests(unittest.TestCase):
    def test_qwen_analyzer_parses_and_sanitizes_local_json(self):
        payload = {
            "candidates": [
                {
                    "start": 20.49,
                    "core_scores": {
                        "immediate_action": 17,
                        "curiosity_gap": 18,
                        "emotional_reaction": 10,
                        "conflict_tension": 13,
                        "visual_surprise": 2,
                        "context_independence": 9,
                        "payoff_proximity": 9,
                    },
                    "human_modifiers": {"naturalness": 8},
                    "penalties": {},
                    "hook_type": "STAKES",
                    "hook_confidence": 0.82,
                }
            ],
            "best_start": 20.49,
        }
        response = {"message": {"content": json.dumps(payload)}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"x")
            with patch("src.hooks.qwen_analyzer.HOOK_CACHE_DIR", root / "cache"), patch(
                "src.hooks.qwen_analyzer.ollama.chat", return_value=response
            ) as chat:
                raw, from_cache = QwenHookAnalyzer().analyze(
                    video_path=video,
                    transcript=[
                        {"start": 20.0, "end": 21.0, "text": "Wait, if this fails we're done"}
                    ],
                    original_start=20.0,
                    highlight_end=35.0,
                    candidate_starts=[19.5, 20.0, 20.5],
                    profile="gaming",
                    video_duration=60.0,
                    clip_index=1,
                )

        self.assertFalse(from_cache)
        self.assertEqual(raw["_source"], "qwen_local_fallback")
        self.assertEqual(raw["candidates"][0]["start"], 20.5)
        self.assertIn("dead_air", raw["candidates"][0]["penalties"])
        self.assertEqual(chat.call_count, 1)

    def test_missing_gemini_key_uses_qwen_instead_of_zero_score_fallback(self):
        fake_local = _FakeLocalAnalyzer()
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"x")
            with patch("src.hooks.analyzer.GEMINI_API_KEY", ""), patch(
                "src.hooks.analyzer.QwenHookAnalyzer", return_value=fake_local
            ), patch("src.hooks.analyzer._load_cache", return_value=None):
                analyzer = GeminiHookAnalyzer()
                raw, from_cache = analyzer.analyze(
                    video_path=video,
                    transcript=[],
                    original_start=20.0,
                    highlight_end=35.0,
                    candidate_starts=[20.0, 20.5],
                    profile="gaming",
                    video_duration=60.0,
                    clip_index=1,
                )

        self.assertFalse(from_cache)
        self.assertEqual(fake_local.calls, 1)
        self.assertEqual(raw["_source"], "qwen_local_fallback")
        self.assertEqual(raw["best_start"], 20.5)


if __name__ == "__main__":
    unittest.main()
