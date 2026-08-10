from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.hooks.candidates import generate_hook_start_candidates
from src.hooks.models import HookOptimizationResult
from src.hooks.optimizer import (
    _apply_optimized_start,
    _attach_metadata,
    optimize_hook_start,
    optimize_hooks,
)
from src.hooks.scoring import (
    calculate_hook_scores,
    clamp_semantic_start,
    normalize_hook_candidate,
    refine_start_locally,
    select_best_hook_candidate,
)


CORE_70 = {
    "immediate_action": 14,
    "curiosity_gap": 14,
    "emotional_reaction": 10,
    "conflict_tension": 10,
    "visual_surprise": 7,
    "context_independence": 8,
    "payoff_proximity": 7,
}

CORE_90 = {
    "immediate_action": 18,
    "curiosity_gap": 18,
    "emotional_reaction": 13,
    "conflict_tension": 14,
    "visual_surprise": 9,
    "context_independence": 9,
    "payoff_proximity": 9,
}


def raw_candidate(
    start: float,
    core=None,
    modifiers=None,
    penalties=None,
    confidence: float = 0.9,
    hook_type: str = "CURIOSITY",
):
    return {
        "start": start,
        "base_hook_score": 0,
        "final_hook_score": 0,
        "core_scores": dict(core or CORE_70),
        "human_modifiers": {
            "contradiction": 0,
            "specificity": 0,
            "timeframe_tension": 0,
            "relatability": 0,
            "naturalness": 0,
            **(modifiers or {}),
        },
        "penalties": dict(penalties or {}),
        "hook_type": hook_type,
        "secondary_hook_types": [],
        "hook_reason": "test",
        "hook_confidence": confidence,
        "loop_score": 0,
    }


class FakeAnalyzer:
    def __init__(self, candidates):
        self.candidates = candidates

    def analyze(self, **_kwargs):
        return {"candidates": self.candidates, "best_start": self.candidates[-1]["start"]}, False


class HookOptimizerV2Tests(unittest.TestCase):
    def test_01_better_start_moves_forward(self):
        highlight = {"start": 20.0, "end": 40.0, "duration": 20.0}
        analyzer = FakeAnalyzer(
            [
                raw_candidate(20.0, CORE_70),
                raw_candidate(21.5, CORE_90, modifiers={"naturalness": 7}),
            ]
        )
        result = optimize_hook_start(
            video_path=Path("missing.mp4"),
            transcript=[],
            highlight=highlight,
            content_profile="gaming",
            analyzer=analyzer,
            video_duration=100.0,
        )
        self.assertTrue(result.applied)
        self.assertAlmostEqual(result.optimized_start, 21.5, places=2)
        self.assertAlmostEqual(highlight["start"], 21.5, places=2)

    def test_02_gemini_failure_uses_original_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = root / "highlights"
            transcript = root / "transcript"
            highlights.mkdir()
            transcript.mkdir()
            (highlights / "video.json").write_text(
                json.dumps([{"start": 20.0, "end": 40.0, "score": 90}]),
                encoding="utf-8",
            )
            (transcript / "video.json").write_text("[]", encoding="utf-8")
            video = root / "video.mp4"
            video.write_bytes(b"x")

            with patch("src.hooks.optimizer.HIGHLIGHTS_DIR", highlights), patch(
                "src.hooks.optimizer.TRANSCRIPT_DIR", transcript
            ), patch("src.hooks.optimizer.probe_duration", return_value=100.0), patch(
                "src.hooks.optimizer.GeminiHookAnalyzer",
                side_effect=RuntimeError("gemini fail"),
            ), patch("src.hooks.optimizer.HOOK_OPTIMIZER_ENABLED", True):
                optimize_hooks("video", video, "gaming")

            saved = json.loads((highlights / "video.json").read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["start"], 20.0)
            self.assertEqual(saved[0]["optimized_start"], 20.0)
            self.assertIn("gemini_unavailable", saved[0]["hook_optimizer"]["fallback_reason"])

    def test_03_low_confidence_is_conservative(self):
        highlight = {"start": 20.0, "end": 40.0}
        analyzer = FakeAnalyzer(
            [
                raw_candidate(20.0, CORE_70, confidence=0.8),
                raw_candidate(21.5, CORE_90, confidence=0.40),
            ]
        )
        result = optimize_hook_start(
            video_path=Path("missing.mp4"),
            transcript=[],
            highlight=highlight,
            content_profile="gaming",
            analyzer=analyzer,
            video_duration=100.0,
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.optimized_start, 20.0)
        self.assertEqual(highlight["start"], 20.0)
        self.assertEqual(result.fallback_reason, "hook_confidence_below_threshold")

    def test_04_max_shift_is_clamped(self):
        self.assertEqual(clamp_semantic_start(99.0, 20.0, 40.0), 23.0)
        candidates = generate_hook_start_candidates(20.0, 40.0, 100.0)
        self.assertGreaterEqual(min(candidates), 18.5)
        self.assertLessEqual(max(candidates), 23.0)

    def test_05_word_clipping_aligns_to_whisper_word(self):
        transcript = [
            {
                "start": 21.3,
                "end": 22.3,
                "text": "wait now",
                "words": [
                    {"word": "wait", "start": 21.3, "end": 21.8},
                    {"word": "now", "start": 21.9, "end": 22.2},
                ],
            }
        ]
        refined = refine_start_locally(21.5, transcript, 20.0, 40.0)
        self.assertLessEqual(refined, 21.3)
        self.assertFalse(21.3 < refined < 21.8)

    def test_06_tie_prefers_naturalness(self):
        a = normalize_hook_candidate(
            raw_candidate(20.0, CORE_70, modifiers={"naturalness": 10})
        )
        b_core = dict(CORE_70)
        b_core["immediate_action"] += 2
        b = normalize_hook_candidate(
            raw_candidate(20.5, b_core, modifiers={"naturalness": -5})
        )
        best = select_best_hook_candidate([a, b], [], 20.0)
        self.assertIs(best, a)

    def test_07_spoiled_payoff_is_penalized(self):
        payoff = normalize_hook_candidate(
            raw_candidate(20.0, CORE_90, penalties={"spoiled_payoff": -30})
        )
        pre_payoff = normalize_hook_candidate(raw_candidate(19.5, CORE_90))
        self.assertGreater(pre_payoff.final_hook_score, payoff.final_hook_score)

    def test_08_disabled_optimizer_preserves_highlight(self):
        highlight = {"start": 20.0, "end": 40.0}
        with patch("src.hooks.optimizer.HOOK_OPTIMIZER_ENABLED", False):
            result = optimize_hook_start(
                video_path=Path("missing.mp4"),
                transcript=[],
                highlight=highlight,
                content_profile="gaming",
                analyzer=FakeAnalyzer([]),
                video_duration=100.0,
            )
        self.assertFalse(result.applied)
        self.assertEqual(highlight, {"start": 20.0, "end": 40.0})

    def test_09_natural_reaction_beats_forced_intro(self):
        forced = normalize_hook_candidate(
            raw_candidate(
                20.0,
                CORE_70,
                modifiers={"naturalness": -13},
                penalties={"forced_intro": -10, "generic_setup": -7},
            )
        )
        natural = normalize_hook_candidate(
            raw_candidate(
                20.5,
                CORE_70,
                modifiers={"naturalness": 8},
                hook_type="NATURAL_REACTION",
            )
        )
        self.assertGreater(natural.final_hook_score, forced.final_hook_score)

    def test_10_specificity_is_a_booster(self):
        generic = normalize_hook_candidate(raw_candidate(20.0, CORE_70))
        specific = normalize_hook_candidate(
            raw_candidate(20.5, CORE_70, modifiers={"specificity": 6})
        )
        self.assertEqual(specific.final_hook_score, generic.final_hook_score + 6)

    def test_11_no_timeframe_is_neutral(self):
        core = dict(CORE_70)
        base, final = calculate_hook_scores(
            core,
            {
                "contradiction": 0,
                "specificity": 0,
                "timeframe_tension": 0,
                "relatability": 0,
                "naturalness": 0,
            },
            {},
        )
        self.assertEqual(final, base)

    def test_12_no_contradiction_is_neutral(self):
        candidate = normalize_hook_candidate(raw_candidate(20.0, CORE_70))
        self.assertEqual(candidate.human_modifiers["contradiction"], 0)
        self.assertEqual(candidate.final_hook_score, candidate.base_hook_score)

    def test_13_positive_modifiers_are_capped(self):
        core_45 = {
            "immediate_action": 10,
            "curiosity_gap": 10,
            "emotional_reaction": 5,
            "conflict_tension": 5,
            "visual_surprise": 5,
            "context_independence": 5,
            "payoff_proximity": 5,
        }
        base, final = calculate_hook_scores(
            core_45,
            {
                "contradiction": 8,
                "specificity": 6,
                "timeframe_tension": 5,
                "relatability": 5,
                "naturalness": 10,
            },
            {},
        )
        self.assertEqual(base, 45)
        self.assertEqual(final, 60)

    def test_14_duration_can_extend_end_only_when_needed(self):
        clip = {"start": 20.0, "end": 32.0, "duration": 12.0}
        applied, reason = _apply_optimized_start(clip, 21.0, 100.0)
        self.assertTrue(applied, reason)
        self.assertEqual(clip["start"], 21.0)
        self.assertEqual(clip["end"], 33.0)
        self.assertEqual(clip["duration"], 12.0)

    def test_15_extract_segments_receive_optimized_start(self):
        clip = {
            "start": 20.0,
            "end": 40.0,
            "segments": [
                {"start": 20.0, "end": 28.0, "role": "hook"},
                {"start": 32.0, "end": 40.0, "role": "payoff"},
            ],
        }
        applied, reason = _apply_optimized_start(clip, 21.5, 100.0)
        self.assertTrue(applied, reason)
        self.assertEqual(clip["segments"][0]["start"], 21.5)
        self.assertEqual(clip["start"], 21.5)

    def test_16_highlight_score_stays_separate_from_hook_score(self):
        clip = {"start": 20.0, "end": 40.0, "score": 95}
        result = HookOptimizationResult(
            original_start=20.0,
            original_end=40.0,
            semantic_start=21.0,
            optimized_start=21.0,
            base_hook_score=58,
            hook_score=62,
            confidence=0.9,
            applied=True,
        )
        _attach_metadata(clip, result)
        self.assertEqual(clip["highlight_score"], 95)
        self.assertEqual(clip["hook_score"], 62)


if __name__ == "__main__":
    unittest.main()
