import unittest
from unittest.mock import patch

from src.editing.smartcut_v3 import _refine_segment


class SmartCutV3Tests(unittest.TestCase):
    def test_cold_open_cannot_expand_into_full_payoff(self):
        with patch(
            "src.editing.smartcut_v3.refine_edit_plan",
            return_value=[{"start": 9.0, "end": 20.0}],
        ):
            result = _refine_segment(
                video_name="video",
                clip_index=1,
                segment_index=1,
                segment={"start": 10.0, "end": 10.8, "role": "cold_open"},
                transcript=[],
                video_duration=30.0,
            )
        self.assertGreaterEqual(result["start"], 9.88)
        self.assertLessEqual(result["end"], 10.98)

    def test_reaction_gets_more_after_window_than_context(self):
        with patch(
            "src.editing.smartcut_v3.refine_edit_plan",
            return_value=[{"start": 10.0, "end": 14.0}],
        ):
            reaction = _refine_segment(
                video_name="video",
                clip_index=1,
                segment_index=1,
                segment={"start": 10.0, "end": 12.0, "role": "reaction"},
                transcript=[],
                video_duration=30.0,
            )
            context = _refine_segment(
                video_name="video",
                clip_index=1,
                segment_index=2,
                segment={"start": 10.0, "end": 12.0, "role": "context"},
                transcript=[],
                video_duration=30.0,
            )
        self.assertGreater(reaction["end"], context["end"])
        self.assertEqual(reaction["end"], 12.5)
        self.assertEqual(context["end"], 12.18)


if __name__ == "__main__":
    unittest.main()
