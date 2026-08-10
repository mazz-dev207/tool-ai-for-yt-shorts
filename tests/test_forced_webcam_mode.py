import unittest
from unittest.mock import patch

import numpy as np

from src.forced_webcam_mode import (
    FORCED_WEBCAM_RATIO_MAX,
    FORCED_WEBCAM_RATIO_MIN,
    _forced_ratio,
    build_forced_gameplay_webcam_plan,
)
from src.gameplay_tracker import GameplayTrackPoint
from src.smart_crop import CropPlan
from src.smart_crop_v2 import SmartCropV2Plan
from src.webcam_detector import WebcamCandidate, WebcamROI


class ForcedWebcamModeTests(unittest.TestCase):
    def _candidate(self, score=0.28):
        return WebcamCandidate(
            roi=WebcamROI(80, 70, 420, 330),
            corner="top-left",
            face_presence_score=0.22,
            face_persistence_score=0.22,
            roi_stability_score=0.94,
            geometric_consistency_score=0.96,
            overlay_likelihood_score=0.78,
            visual_separation_score=0.18,
            temporal_separation_score=0.70,
            false_positive_penalty=0.12,
            final_score=score,
            hits=2,
            total_samples=9,
            sample_indices=[0, 1],
        )

    def _plan(self):
        legacy = CropPlan(
            input_width=1920,
            input_height=1080,
            crop_width=608,
            crop_height=1080,
            duration=10.0,
            points=[],
        )
        return SmartCropV2Plan(
            mode="GENERAL",
            input_width=1920,
            input_height=1080,
            duration=10.0,
            legacy_plan=legacy,
        )

    def test_ratio_contract_stays_between_35_and_40_percent(self):
        normal = _forced_ratio([])
        reaction = _forced_ratio([{"type": "reaction"}, {"type": "reaction"}])
        self.assertGreaterEqual(normal, FORCED_WEBCAM_RATIO_MIN)
        self.assertLessEqual(normal, FORCED_WEBCAM_RATIO_MAX)
        self.assertGreaterEqual(reaction, FORCED_WEBCAM_RATIO_MIN)
        self.assertLessEqual(reaction, FORCED_WEBCAM_RATIO_MAX)
        self.assertGreaterEqual(1.0 - normal, 0.60)
        self.assertLessEqual(1.0 - normal, 0.65)

    def test_forced_mode_accepts_best_candidate_below_auto_threshold(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = [(0.0, frame, []), (0.5, frame, []), (1.0, frame, [])]
        candidate = self._candidate(score=0.28)
        track = [
            GameplayTrackPoint(0.0, 960.0, 540.0, "stable_gameplay_target", {}),
            GameplayTrackPoint(0.5, 980.0, 540.0, "stable_gameplay_target", {}),
            GameplayTrackPoint(1.0, 1000.0, 540.0, "stable_gameplay_target", {}),
        ]
        summary = {
            "target_switches": 0,
            "direction_changes": 0,
            "held_samples": 0,
            "recenter_samples": 0,
            "scene_cuts": 0,
        }

        with patch(
            "src.forced_webcam_mode._sample_video",
            return_value=(1920, 1080, 1.0, samples),
        ), patch(
            "src.forced_webcam_mode.build_webcam_candidates",
            return_value=[candidate],
        ), patch(
            "src.forced_webcam_mode._detect_reaction_events",
            return_value=[],
        ), patch(
            "src.forced_webcam_mode.build_stable_gameplay_track",
            return_value=(track, summary),
        ):
            result = build_forced_gameplay_webcam_plan("dummy.mp4", self._plan())

        self.assertEqual(result.mode, "GAMEPLAY_WEBCAM_STACK")
        self.assertIsNotNone(result.webcam_region)
        self.assertTrue(result.webcam_detection_debug.get("forced"))
        self.assertAlmostEqual(result.layout_metadata["webcam_ratio"], 0.38, places=2)
        self.assertAlmostEqual(result.layout_metadata["gameplay_ratio"], 0.62, places=2)
        self.assertEqual(result.layout_metadata["webcam_position"], "top")
        self.assertEqual(result.layout_metadata["gameplay_position"], "bottom")


if __name__ == "__main__":
    unittest.main()
