import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.caption_hooks.generator import generate_caption_hook
from src.caption_hooks.models import CaptionHookCandidate, CaptionHookResult
from src.caption_hooks.renderer import append_caption_hook_to_ass
from src.caption_hooks.scorer import (
    choose_emphasis,
    score_caption_candidate,
    split_caption_lines,
)


class CaptionHookEngineV3Tests(unittest.TestCase):
    def satisfaction(self, **overrides):
        value = {
            "status": "ok",
            "payoff_score": 92,
            "expectation_match_score": 90,
            "context_independence_score": 88,
            "risks": {
                "missing_context": False,
                "clickbait_gap": False,
            },
            "hook_promise": "One mistake can end the run",
            "actual_payoff": "The player survives the final attempt",
            "payoff": {
                "reason": "The final attempt resolves the run.",
            },
        }
        value.update(overrides)
        return value

    def edit_plan(self, hook_score=70):
        return {
            "hook": {
                "mode": "NATIVE",
                "score": hook_score,
                "text": "",
            },
            "timeline": [
                {
                    "source_start": 10.0,
                    "source_end": 15.0,
                    "purpose": "payoff",
                }
            ],
            "viewer_question": "Can the player survive the final attempt?",
            "stakes": "One mistake ends the run",
            "payoff": "The player survives the final attempt",
            "metadata": {"content_profile": "gaming"},
        }

    def clip(self):
        return {
            "start": 10.0,
            "end": 15.0,
            "title": "Final Attempt",
            "hook": "",
            "payoff": "The player survives the final attempt",
            "satisfaction": self.satisfaction(),
            "v3": {"hook_score_v3": 70},
        }

    def transcript(self):
        return [
            {
                "start": 10.0,
                "end": 12.0,
                "text": "If I miss this the run is over",
                "words": [],
            },
            {
                "start": 12.0,
                "end": 15.0,
                "text": "I actually made it",
                "words": [],
            },
        ]

    def score(self, candidate, **kwargs):
        return score_caption_candidate(
            candidate,
            opening_text=kwargs.get("opening_text", "If I miss this the run is over"),
            payoff_text=kwargs.get("payoff_text", "The player survives the final attempt"),
            truth_context=kwargs.get(
                "truth_context",
                "One mistake ends the run | The player survives the final attempt",
            ),
            satisfaction=kwargs.get("satisfaction", self.satisfaction()),
            content_profile=kwargs.get("content_profile", "gaming"),
        )

    def test_01_strong_valid_caption_hook_scores_high(self):
        candidate = self.score(
            CaptionHookCandidate(
                text="ONE MISTAKE ENDS THE RUN",
                type="stakes",
                mode="COMPLEMENT",
            )
        )
        self.assertGreaterEqual(candidate.score, 75)
        self.assertEqual(candidate.penalties["generic_hype"], 0)
        self.assertEqual(candidate.penalties["unsupported_claim"], 0)

    def test_02_generic_clickbait_is_rejected(self):
        candidate = self.score(
            CaptionHookCandidate(
                text="YOU WON'T BELIEVE WHAT HAPPENS NEXT",
                type="mystery",
                mode="COMPLEMENT",
            ),
            truth_context="The player survives a final attempt",
        )
        self.assertIn("generic_hype", candidate.rejected_reasons)
        self.assertLess(candidate.score, 75)

    def test_03_dialogue_duplication_is_penalized(self):
        opening = "ONE MISTAKE ENDS THE RUN"
        candidate = self.score(
            CaptionHookCandidate(
                text=opening,
                type="stakes",
                mode="REINFORCE",
            ),
            opening_text=opening,
        )
        self.assertIn("dialogue_duplication", candidate.rejected_reasons)
        self.assertGreater(candidate.penalties["dialogue_duplication"], 0)

    def test_04_unsupported_number_is_rejected(self):
        candidate = self.score(
            CaptionHookCandidate(
                text="HE WON BY 1 POINT",
                type="outcome_tease",
                mode="COMPLEMENT",
            ),
            opening_text="The score is close",
            payoff_text="He wins by 2 points",
            truth_context="The score is close | He wins by 2 points",
        )
        self.assertIn("unsupported_number", candidate.rejected_reasons)
        self.assertEqual(candidate.scores["truthfulness"], 0)

    def test_05_exact_payoff_as_mystery_gets_spoiler_penalty(self):
        payoff = "HE FINDS HIM BEHIND THE WALL"
        candidate = self.score(
            CaptionHookCandidate(
                text=payoff,
                type="mystery",
                mode="COMPLEMENT",
            ),
            opening_text="There is one player left",
            payoff_text=payoff,
            truth_context=f"There is one player left | {payoff}",
        )
        self.assertGreater(candidate.penalties["spoiler"], 0)

    def test_06_missing_context_prefers_clarify(self):
        satisfaction = self.satisfaction(
            risks={"missing_context": True, "clickbait_gap": False},
        )
        complement = self.score(
            CaptionHookCandidate(
                text="THIS WAS THE FINAL ROUND",
                type="clarification",
                mode="COMPLEMENT",
            ),
            opening_text="We need this",
            truth_context="This was the final round | We need this",
            satisfaction=satisfaction,
        )
        clarify = self.score(
            CaptionHookCandidate(
                text="THIS WAS THE FINAL ROUND",
                type="clarification",
                mode="CLARIFY",
            ),
            opening_text="We need this",
            truth_context="This was the final round | We need this",
            satisfaction=satisfaction,
        )
        self.assertGreater(clarify.score, complement.score)

    def test_07_gaming_outcome_type_beats_same_caption_in_podcast_profile(self):
        gaming = self.score(
            CaptionHookCandidate(
                text="HE WON THE FINAL ROUND",
                type="outcome_tease",
                mode="COMPLEMENT",
            ),
            opening_text="The final round starts",
            truth_context="He won the final round | The final round starts",
            content_profile="gaming",
        )
        podcast = self.score(
            CaptionHookCandidate(
                text="HE WON THE FINAL ROUND",
                type="outcome_tease",
                mode="COMPLEMENT",
            ),
            opening_text="The final round starts",
            truth_context="He won the final round | The final round starts",
            content_profile="podcast",
        )
        self.assertGreater(gaming.score, podcast.score)

    def test_08_over_max_words_gets_length_penalty(self):
        candidate = self.score(
            CaptionHookCandidate(
                text="THIS IS A VERY LONG CAPTION THAT SHOULD NOT STAY",
                type="mystery",
                mode="COMPLEMENT",
            ),
            truth_context="This is a very long caption that should not stay",
        )
        self.assertGreater(candidate.penalties["too_long"], 0)

    def test_09_two_line_formatter_never_exceeds_two_lines(self):
        lines = split_caption_lines("THE LAST ROUND DECIDES EVERYTHING", 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual(" ".join(lines), "THE LAST ROUND DECIDES EVERYTHING")

    def test_10_emphasis_is_at_most_one_phrase(self):
        emphasis = choose_emphasis("HE WON BY 1 POINT")
        self.assertLessEqual(len(emphasis), 1)
        self.assertEqual(emphasis, ["1 POINT"])

    def test_11_ass_renderer_adds_separate_caption_hook_style(self):
        result = CaptionHookResult(
            enabled=True,
            status="ok",
            text="HE WON BY 1 POINT",
            lines=["HE WON BY", "1 POINT"],
            emphasis=["1 POINT"],
            duration=1.4,
            position="top_safe",
        )
        normal_event = (
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,NORMAL KARAOKE"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip_1.ass"
            path.write_text(
                "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
                "[V4+ Styles]\n"
                "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
                "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
                "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
                "MarginR,MarginV,Encoding\n"
                "Style: Default,Anton,92,&H00FFFFFF,&H0000FFFF,&H00000000,"
                "&H00000000,-1,0,0,0,100,100,0,0,1,4,1,5,120,120,0,1\n\n"
                "[Events]\n"
                "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
                f"{normal_event}\n",
                encoding="utf-8",
            )
            self.assertTrue(append_caption_hook_to_ass(path, result))
            content = path.read_text(encoding="utf-8")
            self.assertIn("Style: CaptionHook,", content)
            self.assertIn("Dialogue: 5,", content)
            self.assertIn(normal_event, content)
            self.assertIn(r"\pos(540,240)", content)

    def test_12_gameplay_webcam_position_stays_below_webcam_safe_zone(self):
        result = CaptionHookResult(
            enabled=True,
            status="ok",
            text="FINAL ROUND",
            lines=["FINAL ROUND"],
            position="gameplay_top",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.ass"
            path.write_text(
                "[V4+ Styles]\n"
                "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
                "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
                "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
                "MarginR,MarginV,Encoding\n"
                "[Events]\n"
                "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n",
                encoding="utf-8",
            )
            append_caption_hook_to_ass(path, result)
            content = path.read_text(encoding="utf-8")
            self.assertIn(r"\pos(540,1160)", content)

    def test_13_disabled_result_does_not_touch_ass(self):
        result = CaptionHookResult(enabled=False, status="none")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.ass"
            original = "[Events]\n"
            path.write_text(original, encoding="utf-8")
            self.assertFalse(append_caption_hook_to_ass(path, result))
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_14_ai_unavailable_continues_without_caption(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.caption_hooks.generator.V3_CAPTION_HOOK_CACHE_DIR", Path(tmp)
        ), patch(
            "src.caption_hooks.generator._generate_local_candidates",
            side_effect=RuntimeError("ollama offline"),
        ), patch(
            "src.caption_hooks.generator._deterministic_candidates",
            return_value=[],
        ):
            result = generate_caption_hook(
                video_name="video",
                clip_index=1,
                clip=self.clip(),
                transcript=self.transcript(),
                edit_plan=self.edit_plan(),
                content_profile="gaming",
            )
        self.assertFalse(result.enabled)
        self.assertEqual(result.status, "unavailable")

    def test_15_score_below_threshold_returns_none(self):
        low = CaptionHookCandidate(
            text="FINAL ATTEMPT",
            type="clarification",
            mode="CLARIFY",
            score=70,
            scores={"clarity": 90},
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.caption_hooks.generator.V3_CAPTION_HOOK_CACHE_DIR", Path(tmp)
        ), patch(
            "src.caption_hooks.generator._generate_local_candidates",
            return_value=[CaptionHookCandidate(text="FINAL ATTEMPT", type="clarification", mode="CLARIFY")],
        ), patch(
            "src.caption_hooks.generator.score_caption_candidate",
            return_value=low,
        ):
            result = generate_caption_hook(
                video_name="video",
                clip_index=1,
                clip=self.clip(),
                transcript=self.transcript(),
                edit_plan=self.edit_plan(hook_score=50),
                content_profile="gaming",
            )
        self.assertFalse(result.enabled)
        self.assertEqual(result.mode, "NONE")

    def test_16_strong_spoken_hook_raises_caption_bar_and_can_choose_none(self):
        medium = CaptionHookCandidate(
            text="FINAL ATTEMPT",
            type="clarification",
            mode="CLARIFY",
            score=82,
            scores={"clarity": 95},
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.caption_hooks.generator.V3_CAPTION_HOOK_CACHE_DIR", Path(tmp)
        ), patch(
            "src.caption_hooks.generator._generate_local_candidates",
            return_value=[CaptionHookCandidate(text="FINAL ATTEMPT", type="clarification", mode="CLARIFY")],
        ), patch(
            "src.caption_hooks.generator.score_caption_candidate",
            return_value=medium,
        ):
            result = generate_caption_hook(
                video_name="video",
                clip_index=1,
                clip=self.clip(),
                transcript=self.transcript(),
                edit_plan=self.edit_plan(hook_score=96),
                content_profile="gaming",
            )
        self.assertFalse(result.enabled)
        self.assertEqual(result.mode, "NONE")

    def test_17_cache_hit_skips_local_regeneration(self):
        high = CaptionHookCandidate(
            text="ONE MISTAKE ENDS THE RUN",
            type="stakes",
            mode="COMPLEMENT",
            score=92,
            scores={"clarity": 96, "truthfulness": 100},
        )
        raw = CaptionHookCandidate(
            text="ONE MISTAKE ENDS THE RUN",
            type="stakes",
            mode="COMPLEMENT",
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.caption_hooks.generator.V3_CAPTION_HOOK_CACHE_DIR", Path(tmp)
        ), patch(
            "src.caption_hooks.generator._generate_local_candidates",
            return_value=[raw],
        ) as first_generator, patch(
            "src.caption_hooks.generator.score_caption_candidate",
            return_value=high,
        ):
            first = generate_caption_hook(
                video_name="video",
                clip_index=1,
                clip=self.clip(),
                transcript=self.transcript(),
                edit_plan=self.edit_plan(hook_score=50),
                content_profile="gaming",
            )
            self.assertTrue(first.enabled)
            self.assertEqual(first_generator.call_count, 1)

            with patch(
                "src.caption_hooks.generator._generate_local_candidates"
            ) as second_generator:
                second = generate_caption_hook(
                    video_name="video",
                    clip_index=1,
                    clip=self.clip(),
                    transcript=self.transcript(),
                    edit_plan=self.edit_plan(hook_score=50),
                    content_profile="gaming",
                )
                second_generator.assert_not_called()
                self.assertTrue(second.cache_hit)

    def test_18_malformed_local_json_falls_back_safely(self):
        clip = self.clip()
        clip["title"] = "FINAL ATTEMPT"
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.caption_hooks.generator.V3_CAPTION_HOOK_CACHE_DIR", Path(tmp)
        ), patch(
            "src.caption_hooks.generator.ollama.chat",
            return_value={"message": {"content": "not json"}},
        ):
            result = generate_caption_hook(
                video_name="video",
                clip_index=1,
                clip=clip,
                transcript=self.transcript(),
                edit_plan=self.edit_plan(hook_score=50),
                content_profile="gaming",
            )
        self.assertIn(result.status, {"ok", "none"})
        self.assertNotEqual(result.status, "unavailable")

    def test_19_generator_off_returns_disabled_without_ai(self):
        with patch(
            "src.caption_hooks.generator.V3_CAPTION_HOOK_GENERATOR", "off"
        ), patch(
            "src.caption_hooks.generator.ollama.chat"
        ) as chat:
            result = generate_caption_hook(
                video_name="video",
                clip_index=1,
                clip=self.clip(),
                transcript=self.transcript(),
                edit_plan=self.edit_plan(),
                content_profile="gaming",
            )
            chat.assert_not_called()
        self.assertFalse(result.enabled)
        self.assertEqual(result.status, "disabled")

    def test_20_ass_renderer_is_idempotent(self):
        result = CaptionHookResult(
            enabled=True,
            status="ok",
            text="FINAL ROUND",
            lines=["FINAL ROUND"],
            emphasis=["ROUND"],
            position="top_safe",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.ass"
            path.write_text(
                "[V4+ Styles]\n"
                "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
                "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
                "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
                "MarginR,MarginV,Encoding\n"
                "[Events]\n"
                "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n",
                encoding="utf-8",
            )
            append_caption_hook_to_ass(path, result)
            append_caption_hook_to_ass(path, result)
            content = path.read_text(encoding="utf-8")
            self.assertEqual(content.count("Style: CaptionHook,"), 1)
            self.assertEqual(content.count("Dialogue: 5,"), 1)


if __name__ == "__main__":
    unittest.main()
