import unittest

from src.satisfaction.models import (
    FinalScoreResult,
    QualityGateResult,
    SatisfactionResult,
)
from src.satisfaction.satisfaction_analyzer import apply_satisfaction_end_optimization
from src.satisfaction.scoring import rank_satisfaction_records
from src.v3.models import EditPlan, HookPlan, TimelineSegment


class SatisfactionRankingEdgeTests(unittest.TestCase):
    def test_mixed_outage_candidate_is_not_discarded_by_other_failed_gate(self):
        failed = SatisfactionResult(
            status="available",
            viewer_satisfaction_score=48,
            payoff_score=38,
            context_independence_score=45,
        ).normalize()
        unavailable = SatisfactionResult(status="unavailable").normalize()
        records = [
            {
                "original_index": 1,
                "clip": {"retention_score": 90},
                "satisfaction": failed,
                "gate": QualityGateResult(
                    status="REJECT",
                    passed=False,
                    reasons=["payoff below gate"],
                ),
                "final_score": FinalScoreResult(score=84.0),
            },
            {
                "original_index": 2,
                "clip": {"retention_score": 82},
                "satisfaction": unavailable,
                "gate": QualityGateResult(
                    status="UNAVAILABLE_FALLBACK",
                    passed=True,
                    reasons=["viewer_satisfaction_unavailable"],
                ),
                "final_score": FinalScoreResult(score=78.0),
            },
        ]

        selected, rejected, mode = rank_satisfaction_records(records)

        self.assertEqual(mode, "unavailable_candidates_preserved")
        self.assertEqual([item["original_index"] for item in selected], [2])
        self.assertEqual([item["original_index"] for item in rejected], [1])

    def test_extended_satisfaction_ending_is_protected_for_smartcut3(self):
        plan = EditPlan(
            clip_index=1,
            hook=HookPlan(
                mode="NATIVE",
                score=90,
                source_start=10.0,
                source_end=12.0,
            ),
            timeline=[
                TimelineSegment(10.0, 16.0, "setup"),
                TimelineSegment(16.0, 20.0, "payoff"),
            ],
        )
        satisfaction = SatisfactionResult(
            status="available",
            viewer_satisfaction_score=90,
            payoff_score=94,
            context_independence_score=88,
            recommended_end=21.0,
        ).normalize()

        changed, _reason = apply_satisfaction_end_optimization(
            plan=plan,
            satisfaction=satisfaction,
            clip={"start": 10.0, "end": 20.0},
            video_duration=100.0,
            context_end=23.0,
        )

        self.assertTrue(changed)
        self.assertEqual(plan.timeline[-1].source_end, 21.0)
        self.assertTrue(
            any(
                item.reason == "satisfaction_selected_ending"
                and item.end == 21.0
                for item in satisfaction.protected_ranges
            )
        )


if __name__ == "__main__":
    unittest.main()
