import unittest

from src.editing.sound_design import build_audio_events


class SoundDesignTests(unittest.TestCase):
    def test_gain_is_safely_capped(self):
        proposal = {"audio_events": [{
            "time": 1.0, "effect": "impact", "intensity": 0.9,
            "duration": 0.2, "gain_db": -2.0, "asset": "impact.wav",
        }]}
        events = build_audio_events(proposal, enabled=True, gain_cap_db=-9.0)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0].gain_db, -9.0)


if __name__ == "__main__":
    unittest.main()
