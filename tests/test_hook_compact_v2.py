from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.hooks.qwen_analyzer import (
    QwenHookAnalyzer,
    _sanitize_payload,
    build_qwen_hook_prompt,
)


class HookCompactV2Tests(unittest.TestCase):
    def test_prompt_requests_only_top_three_compact_candidates(self):
        prompt = build_qwen_hook_prompt(
            original_start=20.0,
            highlight_end=35.0,
            candidate_starts=[18.5, 19.0, 19.5, 20.0, 20.5, 21.0],
            transcript_text="[20.00s-21.00s] Wait, what is that?",
            profile="gaming",
            context_start=18.0,
            context_end=35.5,
        )
        self.assertIn("RETURN ONLY the TOP 3", prompt)
        self.assertIn('"hook_score"', prompt)
        self.assertIn('"confidence"', prompt)
        self.assertNotIn("For EVERY candidate return these fields", prompt)
        self.assertIn("do NOT output all of them", prompt)

    def test_compact_payload_is_limited_to_top_three_and_expanded_locally(self):
        payload = {
            "candidates": [
                {"start": 19.0, "hook_score": 60, "confidence": 0.60, "reason": "setup"},
                {"start": 19.5, "hook_score": 82, "confidence": 0.81, "reason": "reaction begins"},
                {"start": 20.0, "hook_score": 91, "confidence": 0.92, "reason": "strong question"},
                {"start": 20.5, "hook_score": 75, "confidence": 0.72, "reason": "late but clear"},
            ],
            "best_start": 20.0,
        }
        transcript = [
            {"start": 19.5, "end": 20.4, "text": "Wait, what is that?"},
            {"start": 20.4, "end": 21.2, "text": "No, run!"},
        ]
        result = _sanitize_payload(
            payload,
            [19.0, 19.5, 20.0, 20.5],
            transcript=transcript,
            video_path=None,
            original_start=20.0,
            highlight_end=35.0,
        )

        self.assertEqual(result["_mode"], "compact_v2")
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(result["best_start"], 20.0)
        starts = {item["start"] for item in result["candidates"]}
        self.assertEqual(starts, {19.5, 20.0, 20.5})
        for item in result["candidates"]:
            self.assertIn("core_scores", item)
            self.assertIn("human_modifiers", item)
            self.assertIn("penalties", item)
            self.assertIn("hook_confidence", item)
            self.assertIn("_compact_semantic_score", item)
            self.assertIn("_compact_local_signals", item)

    def test_best_start_is_repaired_when_model_points_outside_returned_top_three(self):
        payload = {
            "candidates": [
                {"start": 19.0, "hook_score": 55, "confidence": 0.50, "reason": "weak"},
                {"start": 19.5, "hook_score": 80, "confidence": 0.80, "reason": "good"},
                {"start": 20.0, "hook_score": 90, "confidence": 0.90, "reason": "best"},
                {"start": 20.5, "hook_score": 85, "confidence": 0.85, "reason": "second"},
            ],
            "best_start": 19.0,
        }
        result = _sanitize_payload(
            payload,
            [19.0, 19.5, 20.0, 20.5],
            transcript=[],
            video_path=None,
            original_start=20.0,
            highlight_end=35.0,
        )
        self.assertEqual(result["best_start"], 20.0)
        self.assertNotIn(19.0, {item["start"] for item in result["candidates"]})

    def test_analyzer_uses_small_output_budget_and_local_fallback_after_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"x")
            with patch("src.hooks.qwen_analyzer.HOOK_CACHE_DIR", root / "cache"), patch(
                "src.hooks.qwen_analyzer.ollama.chat",
                return_value={"message": {"content": "{not valid json"}},
            ) as chat:
                result, from_cache = QwenHookAnalyzer().analyze(
                    video_path=video,
                    transcript=[{"start": 20.0, "end": 21.0, "text": "Wait, run!"}],
                    original_start=20.0,
                    highlight_end=35.0,
                    candidate_starts=[19.0, 19.5, 20.0, 20.5, 21.0],
                    profile="gaming",
                    video_duration=60.0,
                    clip_index=1,
                )

        self.assertFalse(from_cache)
        self.assertEqual(chat.call_count, 2)
        first_options = chat.call_args_list[0].kwargs["options"]
        self.assertLessEqual(first_options["num_predict"], 640)
        self.assertEqual(result["_mode"], "compact_v2_local_fallback")
        self.assertGreaterEqual(len(result["candidates"]), 1)
        self.assertLessEqual(len(result["candidates"]), 3)
        self.assertIn(result["best_start"], {item["start"] for item in result["candidates"]})


if __name__ == "__main__":
    unittest.main()
