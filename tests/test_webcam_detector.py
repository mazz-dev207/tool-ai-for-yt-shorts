import unittest

import numpy as np

from src.webcam_detector import build_webcam_candidates, detect_webcam


WIDTH = 1920
HEIGHT = 1080


def frame_with_region(x, y, w, h, value=210):
    frame = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
    frame[y:y + h, x:x + w] = value
    return frame


def samples_for_face(face, count=10, second_faces=None):
    x, y, w, h = face
    samples = []
    for index in range(count):
        frame = frame_with_region(max(0, x - w), max(0, y - h), min(WIDTH - max(0, x - w), w * 3), min(HEIGHT - max(0, y - h), h * 3))
        faces = [(x + (index % 2), y, w, h)]
        if second_faces and index < len(second_faces) and second_faces[index] is not None:
            faces.append(second_faces[index])
        samples.append((index * 0.5, frame, faces))
    return samples


class RobustWebcamDetectorTests(unittest.TestCase):
    def test_persistent_top_left_webcam_wins(self):
        samples = samples_for_face((120, 90, 150, 120), 12)
        result = detect_webcam(samples, WIDTH, HEIGHT)
        self.assertIsNotNone(result.selected)
        self.assertEqual(result.selected.corner, "top-left")
        self.assertGreaterEqual(result.selected.face_persistence_score, 0.95)
        self.assertGreater(result.selected.final_score, 0.58)

    def test_persistent_top_right_webcam_wins(self):
        samples = samples_for_face((1630, 90, 150, 120), 12)
        result = detect_webcam(samples, WIDTH, HEIGHT)
        self.assertIsNotNone(result.selected)
        self.assertEqual(result.selected.corner, "top-right")

    def test_transient_large_center_cutscene_face_is_rejected(self):
        samples = []
        for index in range(12):
            frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
            faces = [(650, 220, 620, 520)] if index in {4, 5, 6} else []
            samples.append((index * 0.5, frame, faces))
        result = detect_webcam(samples, WIDTH, HEIGHT)
        self.assertIsNone(result.selected)

    def test_persistent_webcam_beats_transient_hud_face(self):
        samples = []
        for index in range(12):
            frame = frame_with_region(1450, 0, 470, 360, value=205)
            faces = [(1630 + (index % 2), 90, 150, 120)]
            if index in {2, 5, 8}:
                faces.append((80, 760, 110, 100))
            samples.append((index * 0.5, frame, faces))
        result = detect_webcam(samples, WIDTH, HEIGHT)
        self.assertIsNotNone(result.selected)
        self.assertEqual(result.selected.corner, "top-right")
        self.assertGreaterEqual(len(result.candidates), 2)

    def test_candidates_are_sorted_by_final_score(self):
        samples = []
        for index in range(10):
            frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
            faces = [(140, 100, 150, 120)]
            if index < 6:
                faces.append((1620, 780, 140, 110))
            samples.append((index * 0.5, frame, faces))
        candidates = build_webcam_candidates(samples, WIDTH, HEIGHT)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertGreaterEqual(candidates[0].final_score, candidates[1].final_score)

    def test_debug_scores_are_explicit(self):
        samples = samples_for_face((120, 90, 150, 120), 10)
        result = detect_webcam(samples, WIDTH, HEIGHT)
        self.assertIsNotNone(result.selected)
        debug = result.selected.to_debug_dict()
        for key in (
            "face_presence",
            "face_persistence",
            "roi_stability",
            "geometric_consistency",
            "overlay_likelihood",
            "visual_separation",
            "temporal_separation",
            "false_positive_penalty",
            "final_score",
        ):
            self.assertIn(key, debug)


if __name__ == "__main__":
    unittest.main()
