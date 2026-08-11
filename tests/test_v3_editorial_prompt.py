import unittest

from src.v3.editorial_reasoner import (
    EDITORIAL_SCHEMA,
    _validate_response_shape,
    build_editorial_prompt,
)


class V3EditorialPromptTests(unittest.TestCase):
    def test_prompt_contains_truth_and_editorial_contract(self):
        prompt = build_editorial_prompt(
            clip={"start": 10.0, "end": 20.0, "hook_score": 88},
            transcript_text="[10.0s-12.0s] What is that?",
            profile="gaming",
            context_start=5.0,
            context_end=25.0,
            retry_feedback=["low_context_independence"],
        )
        self.assertIn("RECONSTRUCTED", prompt)
        self.assertIn("No invented facts", prompt)
        self.assertIn("semantic truth", prompt)
        self.assertIn("no_transformation_needed", prompt)
        self.assertIn("low_context_independence", prompt)
        self.assertIn("ABSOLUTE timestamps", prompt)
        self.assertIn("RESTRUCTURED SHORT", prompt)
        self.assertIn("playback_rate MUST be 1.0", prompt)

    def test_schema_requires_structured_editorial_outputs(self):
        required = set(EDITORIAL_SCHEMA["required"])
        self.assertTrue(
            {
                "editorial_angle",
                "originality_analysis",
                "hook_candidates",
                "timeline",
                "context_overlays",
                "visual_events",
                "audio_events",
            }.issubset(required)
        )

    def test_invalid_response_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_response_shape({"timeline": []})


if __name__ == "__main__":
    unittest.main()
