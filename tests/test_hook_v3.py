import unittest

from src.hooks.v3 import select_hook_v3

TRANSCRIPT = [{"start": 10.0, "end": 15.0, "text": "I only have one attempt left. What? No way!", "words": []}]


class HookV3Tests(unittest.TestCase):
    def test_real_reconstructed_quote_can_win(self):
        clip = {"start": 10.0, "end": 20.0, "hook_score": 72, "hook_confidence": 0.8}
        proposal = {"hook_candidates": [{
            "mode": "RECONSTRUCTED", "score": 91, "text": "What? No way!",
            "source_start": 13.0, "source_end": 14.5, "duration": 1.5,
            "reason": "Real reaction before payoff.", "confidence": 0.95,
        }]}
        hook = select_hook_v3(clip=clip, proposal=proposal, transcript=TRANSCRIPT, context_start=8.0, context_end=22.0)
        self.assertEqual(hook.mode, "RECONSTRUCTED")

    def test_editorial_invented_number_is_rejected(self):
        clip = {"start": 10.0, "end": 20.0, "hook_score": 80, "hook_confidence": 0.8}
        proposal = {"hook_candidates": [{
            "mode": "EDITORIAL", "score": 99, "text": "$10,000 was on the line.",
            "source_start": None, "source_end": None, "duration": 1.2,
            "reason": "stakes", "confidence": 0.99,
        }]}
        hook = select_hook_v3(clip=clip, proposal=proposal, transcript=TRANSCRIPT, context_start=8.0, context_end=22.0)
        self.assertEqual(hook.mode, "NATIVE")


if __name__ == "__main__":
    unittest.main()
