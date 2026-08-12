import unittest
from unittest.mock import patch

from src.editing.smartcut_v3 import _protect_refined_segment
from src.satisfaction.final_validator import (
    GeminiFinalSatisfactionValidator,
    _validate_payload,
)
from src.satisfaction.models import (
    FinalScoreResult,
    QualityGateResult,
    SatisfactionResult,
)
from src.satisfaction.satisfaction_analyzer import (
    analyze_viewer_satisfaction,
    apply_satisfaction_end_optimization,
    build_end_candidate_values,
)
from src.satisfaction.scoring import (
    calculate_final_short_score,
    evaluate_quality_gates,
    rank_satisfaction_records,
)
from src.v3.models import EditPlan, HookPlan, OriginalityResult, TimelineSegment


def make_plan(*, hook_score=95, include_reaction=True):
    timeline = [
        TimelineSegment(10.0, 13.0, "setup"),
        TimelineSegment(13.0, 17.0, "escalation"),
        TimelineSegment(17.0, 20.0, "payoff"),
    ]
    if include_reaction:
        timeline.append(TimelineSegment(20.0, 21.0, "reaction"))
    return EditPlan(
        clip_index=1,
        hook=HookPlan(
            mode="NATIVE",
            score=hook_score,
            source_start=10.0,
            source_end=12.0,
            confidence=0.95,
        ),
        timeline=timeline,
        editorial_angle="unexpected reversal",
        payoff="the attempt resolves visibly",
    )


def make_raw(**overrides):
    raw = {
        "viewer_satisfaction_score": 91,
        "payoff_score": 94,
        "expectation_match_score": 92,
        "context_independence_score": 88,
        "clarity_score": 93,
        "emotional_completeness_score": 89,
        "value_density_score": 90,
        "ending_quality_score": 95,
        "hook_promise": "The attempt may fail at the last moment.",
        "actual_payoff": "The attempt fails and the creator reacts.",
        "structure": {
            "setup": True,
            "tension": True,
            "escalation": True,
            "payoff": True,
            "pattern": "setup_tension_payoff_reaction",
        },
        "payoff": {
            "exists": True,
            "type": "unexpected_reaction",
            "timestamp": 18.4,
            "strength": 94,
            "reason": "The visible result resolves the tension established by the opening.",
        },
        "risks": {
            "confusing_start": False,
            "missing_context": False,
            "weak_payoff": False,
            "clickbait_gap": False,
            "clickbait_gap_score": 5,
            "abrupt_ending": False,
            "dead_air": False,
            "generic_outro": False,
        },
        "score_reasons": {
            "payoff_reason": "The result and reaction resolve the attempt.",
            "expectation_match_reason": "The hook promise is fully delivered.",
            "context_independence_reason": "The objective and result are understandable standalone.",
            "clarity_reason": "The subject and action remain clear.",
            "emotional_completeness_reason": "The clip reaches reaction after payoff.",
            "value_density_reason": "No redundant setup is required.",
            "ending_quality_reason": "The clip ends naturally after the reaction.",
        },
        "protected_ranges": [],
        "ending_candidates": [
            {
                "end": 21.0,
                "payoff_score": 94,
                "emotional_completeness_score": 89,
                "ending_quality_score": 95,
                "viewer_satisfaction_score": 91,
                "reason": "Ends after the reaction.",
            }
        ],
        "recommended_end": 21.0,
        "recommended_changes": [],
    }
    for key, value in overrides.items():
        if key == "risks":
            raw["risks"].update(value)
        elif key == "payoff":
            raw["payoff"].update(value)
        elif key == "structure":
            raw["structure"].update(value)
        elif key == "score_reasons":
            raw["score_reasons"].update(value)
        else:
            raw[key] = value
    return raw


def analyze(raw, *, profile="gaming", plan=None, clip=None):
    plan = plan or make_plan()
    clip = clip or {"start": 10.0, "end": 21.0, "retention_score": 90, "hook_score": 95}
    return analyze_viewer_satisfaction(
        clip=clip,
        plan=plan,
        originality=OriginalityResult(
            originality_score=84,
            context_independence=82,
            source_dependency=30,
        ),
        proposal={"viewer_satisfaction": raw},
        content_profile=profile,
        video_duration=100.0,
        context_end=24.0,
    )


class ViewerSatisfactionV3Tests(unittest.TestCase):
    def test_strong_hook_and_strong_payoff_passes(self):
        result = analyze(make_raw())
        gate = evaluate_quality_gates(result)
        final_score = calculate_final_short_score(
            hook_score=97,
            retention_score=90,
            originality_score=84,
            satisfaction=result,
        )
        self.assertTrue(result.available)
        self.assertGreaterEqual(result.viewer_satisfaction_score, 85)
        self.assertEqual(gate.status, "PASS")
        self.assertIsNotNone(final_score.score)
        self.assertGreaterEqual(final_score.score, 85)

    def test_strong_hook_weak_payoff_is_rejected(self):
        raw = make_raw(
            payoff_score=35,
            expectation_match_score=48,
            payoff={"strength": 35},
            risks={"weak_payoff": True, "clickbait_gap": True, "clickbait_gap_score": 58},
        )
        result = analyze(raw)
        gate = evaluate_quality_gates(result)
        final_score = calculate_final_short_score(
            hook_score=97,
            retention_score=92,
            originality_score=86,
            satisfaction=result,
        )
        self.assertEqual(gate.status, "REJECT")
        self.assertTrue(any("payoff" in reason for reason in gate.reasons))
        self.assertTrue(any("clickbait_gap" in reason for reason in gate.reasons))
        self.assertEqual(final_score.components["hook"], 97.0)
        self.assertLess(final_score.score, 90)

    def test_missing_context_fails_context_gate(self):
        result = analyze(
            make_raw(
                context_independence_score=32,
                risks={"missing_context": True, "confusing_start": True},
            )
        )
        gate = evaluate_quality_gates(result)
        self.assertEqual(gate.status, "REJECT")
        self.assertTrue(any("context" in reason for reason in gate.reasons))

    def test_abrupt_ending_is_preserved_as_risk(self):
        result = analyze(
            make_raw(
                ending_quality_score=35,
                emotional_completeness_score=52,
                risks={"abrupt_ending": True},
            )
        )
        self.assertTrue(result.risks.abrupt_ending)
        self.assertEqual(result.ending_quality_score, 35)
        self.assertLess(result.viewer_satisfaction_score, 85)

    def test_gaming_visual_payoff_without_dialogue_is_valid_evidence(self):
        result = analyze(
            make_raw(
                payoff={
                    "exists": True,
                    "type": "gameplay_visual_result",
                    "timestamp": 19.2,
                    "strength": 97,
                    "reason": "The opponent is visibly eliminated even though nobody speaks.",
                },
                payoff_score=97,
            ),
            profile="gaming",
        )
        self.assertTrue(result.payoff.exists)
        self.assertEqual(result.payoff.type, "gameplay_visual_result")
        self.assertEqual(result.payoff.strength, 97)

    def test_podcast_insight_can_be_high_satisfaction_without_action(self):
        result = analyze(
            make_raw(
                payoff={
                    "exists": True,
                    "type": "insight",
                    "strength": 92,
                    "reason": "The explanation resolves the opening claim with a concrete insight.",
                },
                structure={"pattern": "claim_curiosity_explanation_insight"},
                emotional_completeness_score=78,
                clarity_score=96,
                context_independence_score=94,
            ),
            profile="podcast",
        )
        self.assertTrue(result.available)
        self.assertGreaterEqual(result.viewer_satisfaction_score, 85)
        self.assertEqual(result.structure.pattern, "claim_curiosity_explanation_insight")

    def test_funny_reaction_pause_is_protected_from_smartcut(self):
        result = analyze(
            make_raw(
                protected_ranges=[
                    {"start": 14.2, "end": 14.8, "reason": "comedic_pause"}
                ]
            )
        )
        self.assertTrue(any(item.reason == "comedic_pause" for item in result.protected_ranges))

        start, end, reasons = _protect_refined_segment(
            rough_start=14.0,
            rough_end=15.2,
            refined_start=14.5,
            refined_end=15.0,
            protected_ranges=[
                {"start": 14.2, "end": 14.8, "reason": "comedic_pause"}
            ],
            video_duration=100.0,
        )
        self.assertLessEqual(start, 14.2)
        self.assertGreaterEqual(end, 14.8)
        self.assertIn("comedic_pause", reasons)

    def test_real_dead_air_is_not_automatically_protected(self):
        result = analyze(make_raw(risks={"dead_air": True}, protected_ranges=[]))
        self.assertTrue(result.risks.dead_air)
        custom = [item for item in result.protected_ranges if not item.reason.startswith("semantic_")]
        self.assertEqual(custom, [])

    def test_final_validator_rejects_invalid_gemini_json_shape(self):
        with self.assertRaises(ValueError):
            _validate_payload({})
        with self.assertRaises(ValueError):
            _validate_payload("not-json-object")

    def test_final_validator_timeout_is_bounded(self):
        validator = GeminiFinalSatisfactionValidator.__new__(GeminiFinalSatisfactionValidator)
        uploaded = object()
        with self.assertRaises(TimeoutError):
            validator._wait(uploaded, timeout=0.0)

    def test_all_candidates_fail_gates_keeps_best_as_needs_review(self):
        records = []
        for index, score in enumerate((82.0, 76.0, 70.0), start=1):
            satisfaction = SatisfactionResult(
                status="available",
                viewer_satisfaction_score=50,
                payoff_score=40,
                context_independence_score=45,
            ).normalize()
            records.append(
                {
                    "original_index": index,
                    "clip": {"retention_score": 80 - index},
                    "satisfaction": satisfaction,
                    "gate": QualityGateResult(
                        status="REJECT",
                        passed=False,
                        reasons=["payoff below gate"],
                        needs_review=True,
                    ),
                    "final_score": FinalScoreResult(score=score),
                }
            )

        with patch("src.satisfaction.scoring.V3_SATISFACTION_FALLBACK_TOP_K", 2):
            selected, rejected, mode = rank_satisfaction_records(records)

        self.assertEqual(mode, "all_failed_keep_best_for_review")
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(rejected), 1)
        self.assertTrue(all(item["gate"].status == "NEEDS_REVIEW" for item in selected))

    def test_satisfaction_unavailable_preserves_upstream_order_and_renormalizes(self):
        unavailable = SatisfactionResult(status="unavailable").normalize()
        final_score = calculate_final_short_score(
            hook_score=80,
            retention_score=70,
            originality_score=60,
            satisfaction=unavailable,
        )
        self.assertIsNotNone(final_score.score)
        self.assertNotIn("satisfaction", final_score.components)
        self.assertIn("satisfaction", final_score.unavailable_components)
        self.assertAlmostEqual(sum(final_score.effective_weights.values()), 1.0, places=4)

        records = [
            {
                "original_index": 2,
                "clip": {"retention_score": 70},
                "satisfaction": unavailable,
                "gate": QualityGateResult(status="UNAVAILABLE_FALLBACK", passed=True),
                "final_score": FinalScoreResult(score=65.0),
            },
            {
                "original_index": 1,
                "clip": {"retention_score": 90},
                "satisfaction": unavailable,
                "gate": QualityGateResult(status="UNAVAILABLE_FALLBACK", passed=True),
                "final_score": FinalScoreResult(score=90.0),
            },
        ]
        selected, rejected, mode = rank_satisfaction_records(records)
        self.assertEqual(mode, "satisfaction_unavailable_preserve_order")
        self.assertEqual([item["original_index"] for item in selected], [2, 1])
        self.assertEqual(rejected, [])

    def test_end_candidates_match_requested_offsets_and_stay_in_bounds(self):
        values = build_end_candidate_values(20.0, 22.0, 100.0)
        self.assertEqual(values, [19.5, 20.0, 20.5, 21.0, 22.0])

    def test_satisfaction_can_extend_end_for_reaction(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(mode="NATIVE", score=90, source_start=10.0, source_end=12.0),
            timeline=[
                TimelineSegment(10.0, 16.0, "setup"),
                TimelineSegment(16.0, 20.0, "payoff"),
            ],
        )
        result = analyze(
            make_raw(
                recommended_end=21.0,
                ending_candidates=[
                    {
                        "end": 20.0,
                        "payoff_score": 80,
                        "emotional_completeness_score": 55,
                        "ending_quality_score": 55,
                        "viewer_satisfaction_score": 60,
                        "reason": "Cuts before reaction.",
                    },
                    {
                        "end": 21.0,
                        "payoff_score": 94,
                        "emotional_completeness_score": 92,
                        "ending_quality_score": 96,
                        "viewer_satisfaction_score": 93,
                        "reason": "Includes reaction.",
                    },
                ],
            ),
            plan=plan,
            clip={"start": 10.0, "end": 20.0, "retention_score": 88, "hook_score": 90},
        )
        changed, _reason = apply_satisfaction_end_optimization(
            plan=plan,
            satisfaction=result,
            clip={"start": 10.0, "end": 20.0},
            video_duration=100.0,
            context_end=23.0,
        )
        self.assertTrue(changed)
        self.assertEqual(plan.timeline[-1].source_end, 21.0)
        self.assertTrue(result.end_adjustment_applied)


if __name__ == "__main__":
    unittest.main()
