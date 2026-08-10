import unittest

from src.hooks.analyzer import build_hook_prompt


class HookPromptTests(unittest.TestCase):
    def test_prompt_contains_four_question_test_and_neutral_modifiers(self):
        prompt = build_hook_prompt(
            original_start=20.0,
            highlight_end=40.0,
            candidate_starts=[19.5, 20.0, 20.5],
            transcript_text="[20.00s-21.00s] Wait... what is that?",
            profile="gaming",
            context_start=18.5,
            context_end=40.5,
        )
        self.assertIn("WHAT IS HAPPENING?", prompt)
        self.assertIn("WHY SHOULD I CARE?", prompt)
        self.assertIn("WHAT HAPPENS NEXT?", prompt)
        self.assertIn("DOES THIS FEEL REAL?", prompt)
        self.assertIn("DO NOT penalize its absence", prompt)
        self.assertIn("naturalness: -15 to +10", prompt)
        self.assertIn("Highlight Score != Hook Score", prompt)


if __name__ == "__main__":
    unittest.main()
