import unittest

from src.story.restructurer import restructure_story


class StoryRestructurerTests(unittest.TestCase):
    def test_reordered_story_is_preserved(self):
        clip = {"start": 10.0, "end": 20.0}
        proposal = {"timeline": [
            {"source_start": 18.0, "source_end": 19.0, "purpose": "cold_open"},
            {"source_start": 10.0, "source_end": 13.0, "purpose": "context"},
            {"source_start": 14.0, "source_end": 17.0, "purpose": "escalation"},
            {"source_start": 18.0, "source_end": 20.0, "purpose": "payoff"},
        ]}
        segments, debug = restructure_story(clip=clip, proposal=proposal, source_duration=30.0, enable_restructuring=True)
        self.assertTrue(debug["reordered"])
        self.assertEqual(segments[0].purpose, "cold_open")
        self.assertEqual(segments[1].source_start, 10.0)

    def test_invalid_timeline_falls_back(self):
        clip = {"start": 10.0, "end": 20.0}
        proposal = {"timeline": [{"source_start": 50.0, "source_end": 55.0, "purpose": "payoff"}]}
        segments, debug = restructure_story(clip=clip, proposal=proposal, source_duration=30.0, enable_restructuring=True)
        self.assertIsNotNone(debug["fallback"])
        self.assertEqual(segments[0].source_start, 10.0)


if __name__ == "__main__":
    unittest.main()
