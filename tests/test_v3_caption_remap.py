import unittest

from src.retention.timeline import extract_remapped_words


class V3CaptionRemapTests(unittest.TestCase):
    def test_words_follow_editorial_segment_order(self):
        transcript = [
            {
                "start": 10.0,
                "end": 11.0,
                "text": "context",
                "words": [{"word": "context", "start": 10.0, "end": 10.5}],
            },
            {
                "start": 18.0,
                "end": 19.0,
                "text": "what payoff",
                "words": [
                    {"word": "what", "start": 18.0, "end": 18.3},
                    {"word": "payoff", "start": 18.5, "end": 18.9},
                ],
            },
        ]
        segments = [
            {"start": 18.0, "end": 18.35, "role": "cold_open"},
            {"start": 10.0, "end": 10.6, "role": "context"},
            {"start": 18.45, "end": 19.0, "role": "payoff"},
        ]
        words = extract_remapped_words(transcript, segments)
        self.assertEqual([word["word"] for word in words], ["what", "context", "payoff"])
        self.assertLess(words[0]["start"], words[1]["start"])
        self.assertLess(words[1]["start"], words[2]["start"])


if __name__ == "__main__":
    unittest.main()
