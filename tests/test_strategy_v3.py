import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.highlights.gemini_judge import build_prompt
from src.highlights.profiles import get_profile
from src.strategy.experiments import export_experiment_metadata
from src.strategy.format_intelligence import (
    apply_format_intelligence,
    calculate_content_opportunity,
    choose_best_format,
    score_format_match,
)
from src.strategy.models import ChannelStrategy, FormatBrief, FormatProfile


FORMAT = FormatProfile(
    id="gaming_high_stakes_fail_reaction",
    niche="gaming",
    subniche="streamer_moments",
    format_name="High-Stakes Fail + Reaction",
    description="Clear attempt, stakes, failure and authentic reaction.",
    required_signals=["clear_goal", "stakes_or_risk", "failure_or_reversal", "reaction"],
    preferred_hook_types=["native", "reconstructed"],
    preferred_story_shape=["cold_open", "compressed_context", "escalation", "payoff", "reaction"],
    target_duration_seconds=(12.0, 28.0),
)

STRATEGY = ChannelStrategy(
    channel_name="MazClips Gaming",
    primary_niche="gaming",
    positioning="fast self-contained gaming moments with clear stakes and authentic payoff",
    priority_formats=[FORMAT.id],
    min_format_fit=45,
    min_content_opportunity=50,
)


def strong_candidate():
    return {
        "start": 10.0,
        "end": 28.0,
        "title": "Last attempt goes wrong",
        "hook": "I only have one attempt left and need to win this",
        "payoff": "I failed at the last second and yelled no way",
        "reason": "clear challenge with immediate reaction",
        "scores": {
            "hook": 88,
            "payoff": 91,
            "standalone_context": 82,
            "retention": 86,
            "emotional_intensity": 90,
            "entertainment": 87,
            "viral_potential": 85,
            "pacing": 78,
        },
    }


class StrategyV3Tests(unittest.TestCase):
    def test_format_match_detects_required_signals(self):
        transcript = [{"start": 10, "end": 28, "text": "Last chance. I need to win. Oh no, I failed. NO WAY!"}]
        match = score_format_match(strong_candidate(), transcript, FORMAT, STRATEGY, None)
        self.assertGreaterEqual(match.format_fit_score, 70)
        self.assertEqual(match.missing_signals, [])
        self.assertEqual(match.recommendation, "use")

    def test_content_opportunity_is_separate_score(self):
        transcript = [{"start": 10, "end": 28, "text": "One attempt left, then I fail and react."}]
        match = score_format_match(strong_candidate(), transcript, FORMAT, STRATEGY, None)
        opportunity = calculate_content_opportunity(strong_candidate(), FORMAT, match, STRATEGY, None)
        self.assertGreater(opportunity.total, 0)
        self.assertLessEqual(opportunity.total, 100)
        self.assertIn("format_match", opportunity.components)
        self.assertNotEqual(opportunity.total, strong_candidate()["scores"]["viral_potential"])

    def test_manual_format_selection(self):
        other = FormatProfile(
            id="gaming_visual_surprise",
            niche="gaming",
            subniche="streamer_moments",
            format_name="Visual Surprise",
            description="Unexpected visual event",
            required_signals=["visual_surprise", "reaction_or_consequence"],
        )
        profile, match, all_matches = choose_best_format(
            strong_candidate(), [], [FORMAT, other], STRATEGY, None,
            manual_format_id=FORMAT.id,
        )
        self.assertEqual(profile.id, FORMAT.id)
        self.assertEqual(len(all_matches), 1)
        self.assertEqual(match.format_id, FORMAT.id)

    def test_external_demand_is_not_added_to_opportunity_score(self):
        brief = FormatBrief(priority_formats=[FORMAT.id], external_demand_signal=99)
        match = score_format_match(strong_candidate(), [], FORMAT, STRATEGY, brief)
        without = calculate_content_opportunity(strong_candidate(), FORMAT, match, STRATEGY, None)
        with_demand = calculate_content_opportunity(strong_candidate(), FORMAT, match, STRATEGY, brief)
        self.assertEqual(without.total, with_demand.total)
        self.assertEqual(with_demand.external_demand_signal, 99)

    def test_apply_format_intelligence_annotates_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = root / "highlights"
            transcript_dir = root / "transcript"
            highlights.mkdir()
            transcript_dir.mkdir()
            (highlights / "video.json").write_text(json.dumps([strong_candidate()]), encoding="utf-8")
            (transcript_dir / "video.json").write_text(
                json.dumps([{"start": 10, "end": 28, "text": "One attempt left. I failed. No way!"}]),
                encoding="utf-8",
            )
            with patch("src.strategy.format_intelligence.HIGHLIGHTS_DIR", highlights), patch(
                "src.strategy.format_intelligence.TRANSCRIPT_DIR", transcript_dir
            ), patch(
                "src.strategy.format_intelligence.resolve_strategy_path", return_value=None
            ), patch(
                "src.strategy.format_intelligence.build_fallback_strategy", return_value=STRATEGY
            ), patch(
                "src.strategy.format_intelligence.load_format_library", return_value=[FORMAT]
            ):
                apply_format_intelligence("video", "gaming")
            saved = json.loads((highlights / "video.json").read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["format_id"], FORMAT.id)
            self.assertIn("content_opportunity_score", saved[0])
            self.assertIn("strategy_context", saved[0])

    def test_gemini_prompt_contains_format_strategy(self):
        candidate = strong_candidate()
        candidate["strategy_context"] = {
            "channel_strategy": STRATEGY.to_dict(),
            "format_profile": FORMAT.to_dict(),
            "format_match": {"format_fit_score": 88, "missing_signals": []},
        }
        prompt = build_prompt(1, candidate, "[10-28] transcript", get_profile("gaming"), 8, 30)
        self.assertIn("FORMAT / CHANNEL STRATEGY CONTEXT", prompt)
        self.assertIn(FORMAT.id, prompt)
        self.assertIn("Do NOT invent missing stakes", prompt)
        self.assertIn("separate strategic signals", prompt)

    def test_experiment_export_creates_per_short_strategy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = root / "highlights"
            output = root / "output"
            highlights.mkdir()
            clip = strong_candidate()
            clip.update({
                "format_id": FORMAT.id,
                "format_version": 1,
                "format_fit_score": 88,
                "content_opportunity_score": 84,
                "strategic_gate": "PASS",
                "duration": 18,
                "v3": {"hook_mode": "RECONSTRUCTED", "hook_score_v3": 90, "originality_score": 79},
                "strategy_context": {
                    "channel_strategy": STRATEGY.to_dict(),
                    "format_profile": FORMAT.to_dict(),
                    "format_match": {"format_fit_score": 88},
                    "content_opportunity": {"total": 84},
                },
            })
            (highlights / "video.json").write_text(json.dumps([clip]), encoding="utf-8")
            with patch("src.strategy.experiments.HIGHLIGHTS_DIR", highlights), patch(
                "src.strategy.experiments.OUTPUT_DIR", output
            ):
                export_experiment_metadata("video")
            debug = output / "debug" / "video" / "clip_1"
            self.assertTrue((debug / "format_profile.json").exists())
            self.assertTrue((debug / "format_match.json").exists())
            self.assertTrue((debug / "content_opportunity.json").exists())
            exported = json.loads((debug / "experiment_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(exported["format_id"], FORMAT.id)
            self.assertEqual(exported["hook_type"], "RECONSTRUCTED")


if __name__ == "__main__":
    unittest.main()
