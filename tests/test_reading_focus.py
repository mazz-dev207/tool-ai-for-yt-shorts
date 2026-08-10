import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.reading_focus import (
    ReadingTarget,
    apply_reading_focus,
    detect_text_candidates,
)
from src.smart_crop_v2 import FocusPoint, Rect


class ReadingFocusTests(unittest.TestCase):
    def test_detects_large_readable_text_region(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(frame, (420, 300), (1510, 610), (25, 25, 25), -1)
        cv2.putText(
            frame,
            "READ THIS MESSAGE",
            (500, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "SECOND LINE HERE",
            (520, 535),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (240, 240, 240),
            4,
            cv2.LINE_AA,
        )

        candidates = detect_text_candidates(frame)
        self.assertTrue(candidates)
        best = candidates[0]
        self.assertGreater(best.w, 300)
        self.assertGreater(best.confidence, 0.34)

    def test_ignored_webcam_region_is_not_reading_target(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "STREAMER NAME",
            (70, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        ignored = Rect(0, 0, 620, 380)
        candidates = detect_text_candidates(frame, ignored_region=ignored)
        for candidate in candidates:
            self.assertFalse(candidate.x < ignored.x2 and candidate.y < ignored.y2)

    def test_reading_focus_moves_camera_toward_persistent_text(self):
        focus_points = [
            FocusPoint(0.0, 960.0, 540.0, "stable_gameplay_target"),
            FocusPoint(0.5, 960.0, 540.0, "stable_gameplay_target"),
            FocusPoint(1.0, 960.0, 540.0, "stable_gameplay_target"),
        ]
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = [(0.0, frame, []), (0.5, frame, []), (1.0, frame, [])]
        targets = [
            None,
            ReadingTarget(0.5, 1450.0, 360.0, 0.80, (1250, 280, 400, 160), 2, 0.90),
            ReadingTarget(1.0, 1450.0, 360.0, 0.86, (1250, 280, 400, 160), 3, 0.92),
        ]

        with patch("src.reading_focus.detect_reading_targets", return_value=targets):
            adjusted, summary = apply_reading_focus(
                focus_points,
                samples,
                crop_width=700,
                crop_height=1080,
                frame_width=1920,
                frame_height=1080,
            )

        self.assertEqual(len(adjusted), 3)
        self.assertEqual(adjusted[1].source, "reading_focus")
        self.assertGreater(adjusted[1].center_x, 960.0)
        self.assertGreater(adjusted[2].center_x, adjusted[1].center_x)
        self.assertEqual(summary["reading_focus_events"], 1)
        self.assertEqual(summary["reading_focus_samples"], 2)

    def test_release_back_to_gameplay_is_gradual(self):
        focus_points = [
            FocusPoint(0.0, 960.0, 540.0, "stable_gameplay_target"),
            FocusPoint(0.5, 960.0, 540.0, "stable_gameplay_target"),
            FocusPoint(1.0, 960.0, 540.0, "stable_gameplay_target"),
        ]
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        samples = [(0.0, frame, []), (0.5, frame, []), (1.0, frame, [])]
        targets = [
            ReadingTarget(0.0, 1450.0, 360.0, 0.85, (1250, 280, 400, 160), 3, 0.90),
            None,
            None,
        ]

        with patch("src.reading_focus.detect_reading_targets", return_value=targets):
            adjusted, _summary = apply_reading_focus(
                focus_points,
                samples,
                crop_width=700,
                crop_height=1080,
                frame_width=1920,
                frame_height=1080,
            )

        self.assertGreater(adjusted[0].center_x, 960.0)
        self.assertGreater(adjusted[1].center_x, 960.0)
        self.assertLess(adjusted[1].center_x, adjusted[0].center_x)
        self.assertLess(adjusted[2].center_x, adjusted[1].center_x)


if __name__ == "__main__":
    unittest.main()
