from __future__ import annotations

import unittest

from src.caption_hooks.models import CaptionHookCandidate
from src.caption_hooks.scorer import score_caption_candidate


SATISFACTION = {
    "status": "ok",
    "payoff_score": 90,
    "expectation_match_score": 90,
    "context_independence_score": 90,
    "risks": {"missing_context": False},
}


def score(
    text: str,
    *,
    hook_type: str,
    opening: str = "WE HAVE ONE TRY LEFT",
    payoff: str = "THE PLAYER SURVIVES THE FINAL ATTEMPT",
    truth: str | None = None,
):
    truth = truth or (
        f"{opening} | ONE MISTAKE ENDS THE RUN | "
        f"{payoff} | BISON BREAKS CHAOS ENSUES"
    )
    return score_caption_candidate(
        CaptionHookCandidate(
            text=text,
            type=hook_type,
            mode="COMPLEMENT",
        ),
        opening_text=opening,
        payoff_text=payoff,
        truth_context=truth,
        satisfaction=SATISFACTION,
        content_profile="gaming",
    )


class CaptionHookQualityV3Tests(unittest.TestCase):
    def test_near_verbatim_dialogue_duplication_is_effectively_rejected(self):
        opening = "IT'S BREAKING. IT'S BREAKING"
        candidate = score(
            opening,
            hook_type="reaction_setup",
            opening=opening,
            truth=f"{opening} | the object is breaking during the attempt",
        )
        self.assertIn("dialogue_duplication", candidate.rejected_reasons)
        self.assertGreaterEqual(candidate.penalties["dialogue_duplication"], 60.0)
        self.assertLess(candidate.score, 75.0)

    def test_overused_chaos_language_is_penalized(self):
        candidate = score(
            "BISON BREAKS CHAOS ENSUES",
            hook_type="unexpected_discovery",
            opening="THE BISON IS STUCK",
            truth="The bison breaks free and chaos ensues",
        )
        self.assertGreater(candidate.penalties["overused_language"], 0.0)

    def test_insane_crazy_and_unbelievable_are_penalized(self):
        for text in (
            "THIS TURN IS INSANE",
            "THIS TURN GETS CRAZY",
            "AN UNBELIEVABLE FINAL TURN",
        ):
            candidate = score(
                text,
                hook_type="mystery",
                opening="THE FINAL TURN STARTS",
                truth=f"The final turn starts | {text}",
            )
            self.assertGreater(
                candidate.penalties["overused_language"],
                0.0,
                msg=text,
            )

    def test_tension_type_beats_flat_description_for_same_truth(self):
        text = "ONE MISTAKE ENDS THE RUN"
        stakes = score(
            text,
            hook_type="stakes",
            opening="THE FINAL ATTEMPT STARTS",
            truth=f"The final attempt starts | {text}",
        )
        flat = score(
            text,
            hook_type="clarification",
            opening="THE FINAL ATTEMPT STARTS",
            truth=f"The final attempt starts | {text}",
        )
        self.assertGreater(stakes.scores["tension"], flat.scores["tension"])
        self.assertGreater(stakes.score, flat.score)

    def test_exact_payoff_outcome_tease_still_gets_spoiler_penalty(self):
        payoff = "PLAYER SURVIVES FINAL ATTEMPT"
        candidate = score(
            payoff,
            hook_type="outcome_tease",
            opening="ONE TRY LEFT",
            payoff=payoff,
            truth=f"One try left | {payoff}",
        )
        self.assertGreater(candidate.penalties["spoiler"], 0.0)


if __name__ == "__main__":
    unittest.main()
