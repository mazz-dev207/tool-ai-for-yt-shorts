import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.hooks.local_refinement import refine_start_multimodally


class HookLocalRefinementTests(unittest.TestCase):
    def test_missing_video_falls_back_to_text_boundary(self):
        transcript = [
            {
                "start": 20.0,
                "end": 21.0,
                "text": "wait what",
                "words": [
                    {"word": "wait", "start": 20.0, "end": 20.35},
                    {"word": "what", "start": 20.45, "end": 20.80},
                ],
            }
        ]
        result = refine_start_multimodally(
            video_path=Path("definitely_missing_video.mp4"),
            semantic_start=20.18,
            transcript=transcript,
            original_start=20.0,
            highlight_end=40.0,
        )
        self.assertFalse(20.0 < result < 20.35)
        self.assertGreaterEqual(result, 18.5)
        self.assertLessEqual(result, 23.0)

    def test_audio_visual_signals_only_polish_local_boundary(self):
        with patch(
            "src.hooks.local_refinement.refine_start_locally",
            return_value=20.2,
        ), patch(
            "src.hooks.local_refinement._candidate_times",
            return_value=[20.2, 20.5],
        ), patch(
            "src.hooks.local_refinement._extract_audio",
            return_value=(np.ones(16000, dtype=np.float32) * 0.01, 19.8),
        ), patch(
            "src.hooks.local_refinement._audio_boundary_score",
            side_effect=lambda _signal, _start, candidate: 1.0 if candidate == 20.2 else 0.0,
        ), patch(
            "src.hooks.local_refinement._visual_and_reaction_score",
            side_effect=lambda _path, candidate: (1.0, 1.0) if candidate == 20.2 else (0.0, 0.0),
        ):
            result = refine_start_multimodally(
                video_path=Path("missing.mp4"),
                semantic_start=20.5,
                transcript=[],
                original_start=20.0,
                highlight_end=40.0,
            )
        self.assertEqual(result, 20.2)
        self.assertLessEqual(abs(result - 20.5), 0.35)


if __name__ == "__main__":
    unittest.main()
