import unittest

from src.hooks.scoring import normalize_hook_candidate, select_best_hook_candidate


CORE_70 = {
    "immediate_action": 14,
    "curiosity_gap": 14,
    "emotional_reaction": 10,
    "conflict_tension": 10,
    "visual_surprise": 7,
    "context_independence": 8,
    "payoff_proximity": 7,
}


def raw(start, core, naturalness):
    return {
        "start": start,
        "core_scores": core,
        "human_modifiers": {
            "contradiction": 0,
            "specificity": 0,
            "timeframe_tension": 0,
            "relatability": 0,
            "naturalness": naturalness,
        },
        "penalties": {},
        "hook_type": "CURIOSITY",
        "secondary_hook_types": [],
        "hook_reason": "tie test",
        "hook_confidence": 0.9,
        "loop_score": 0,
    }


class HookTieBreakTests(unittest.TestCase):
    def test_close_final_scores_prefer_naturalness(self):
        # Both candidates are equally distant from original_start=20.0, so the
        # tie is not decided by shift distance.
        # A: base 70 + naturalness 5 = 75.
        a = normalize_hook_candidate(raw(19.5, dict(CORE_70), 5))

        # B: base 77 + modifiers 0 = 77. Difference=2 <= tie threshold 3.
        b_core = dict(CORE_70)
        b_core["immediate_action"] += 7
        b = normalize_hook_candidate(raw(20.5, b_core, 0))

        self.assertEqual(a.final_hook_score, 75)
        self.assertEqual(b.final_hook_score, 77)
        best = select_best_hook_candidate([a, b], [], 20.0)
        self.assertIs(best, a)


if __name__ == "__main__":
    unittest.main()
