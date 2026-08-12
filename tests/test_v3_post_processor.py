import unittest

from src.editing.post_processor import (
    _build_video_filtergraph,
    _dialogue_duck_expression,
)


class V3PostProcessorTests(unittest.TestCase):
    def test_punch_in_uses_real_pixel_overlay_graph(self):
        plan = {
            "visual_events": [
                {
                    "time": 1.0,
                    "effect": "punch_in",
                    "intensity": 0.8,
                    "duration": 0.5,
                    "target": "center",
                    "metadata": {"semantic_event": "emphasis"},
                }
            ]
        }
        graph, label = _build_video_filtergraph(plan, None)
        self.assertIsNotNone(graph)
        self.assertEqual(label, "v3e0out")
        self.assertIn("[0:v]split=2", graph)
        self.assertIn("scale=", graph)
        self.assertIn("crop=1080:1920", graph)
        self.assertIn("overlay=0:0", graph)
        self.assertIn("between(t,1.000,1.500)", graph)

    def test_multiple_semantic_visual_events_chain(self):
        plan = {
            "visual_events": [
                {
                    "time": 1.0,
                    "effect": "punch_in",
                    "intensity": 0.7,
                    "duration": 0.4,
                },
                {
                    "time": 3.0,
                    "effect": "focus_crop",
                    "intensity": 0.6,
                    "duration": 0.5,
                },
            ]
        }
        graph, label = _build_video_filtergraph(plan, None)
        self.assertEqual(label, "v3e1out")
        self.assertIn("[v3e0out]split=2", graph)
        self.assertIn("between(t,3.000,3.500)", graph)

    def test_face_zoom_is_supported_as_semantic_effect(self):
        plan = {
            "visual_events": [
                {
                    "time": 2.0,
                    "effect": "face_zoom",
                    "intensity": 0.8,
                    "duration": 0.6,
                }
            ]
        }
        graph, label = _build_video_filtergraph(plan, None)
        self.assertEqual(label, "v3e0out")
        self.assertIn("between(t,2.000,2.600)", graph)

    def test_unsupported_effect_is_not_executed(self):
        plan = {
            "visual_events": [
                {
                    "time": 1.0,
                    "effect": "freeze_frame",
                    "intensity": 1.0,
                    "duration": 0.5,
                }
            ]
        }
        graph, label = _build_video_filtergraph(plan, None)
        self.assertIsNone(graph)
        self.assertIsNone(label)

    def test_dialogue_ducking_only_covers_sfx_windows(self):
        expression = _dialogue_duck_expression([(1.0, 1.4), (3.0, 3.3)])
        self.assertIn("between(t,1.000,1.400)", expression)
        self.assertIn("between(t,3.000,3.300)", expression)
        self.assertIn("0.820", expression)
        self.assertTrue(expression.endswith("1))"))


if __name__ == "__main__":
    unittest.main()
