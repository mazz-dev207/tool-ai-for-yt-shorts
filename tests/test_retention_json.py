import unittest
from unittest.mock import patch

from src.retention.analyzer import RetentionAnalyzer, parse_retention_json


class RetentionJsonTests(unittest.TestCase):
    def test_parses_json_code_fence(self):
        result = parse_retention_json('```json\n{"content_type":"gaming"}\n```')
        self.assertEqual(result["content_type"], "gaming")

    def test_repairs_trailing_commas(self):
        result = parse_retention_json(
            '{"content_type":"gaming","scores":{"hook":80,},}'
        )
        self.assertEqual(result["scores"]["hook"], 80)

    @patch("src.retention.analyzer.ollama.chat")
    def test_truncated_response_retries_once_with_compact_json(self, mock_chat):
        mock_chat.side_effect = [
            {
                "message": {"content": '{"content_type":"entertainment","scores":{"hook":80'},
                "load_duration": 0,
                "prompt_eval_count": 1000,
                "eval_count": 800,
            },
            {
                "message": {
                    "content": (
                        '{"content_type":"entertainment",'
                        '"scores":{"hook":80,"curiosity":75,"emotion":60,"conflict":20,'
                        '"payoff":78,"information_density":70,"pacing":76,"standalone":82},'
                        '"hook_variants":[],"variants":[{"name":"best_edit",'
                        '"strategy":"extractive","segments":[{"start":10.0,"end":30.0,'
                        '"role":"hook"}]}]}'
                    )
                },
                "load_duration": 0,
                "prompt_eval_count": 1000,
                "eval_count": 210,
            },
        ]

        analyzer = RetentionAnalyzer(model="test-model")
        result = analyzer.analyze(
            {
                "candidate_start": 10.0,
                "candidate_end": 30.0,
                "start": 5.0,
                "end": 35.0,
                "segments": [
                    {"start": 10.0, "end": 30.0, "text": "A compact vlog story happens here."}
                ],
            },
            {"word_rate": 2.0},
        )

        self.assertEqual(mock_chat.call_count, 2)
        self.assertEqual(result["content_type"], "entertainment")
        self.assertEqual(result["variants"][0]["segments"][0]["end"], 30.0)
        self.assertEqual(result["retention_anchors"], [])

    @patch("src.retention.analyzer.ollama.chat")
    def test_valid_first_response_does_not_retry(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": (
                    '{"content_type":"gaming","scores":{},"hook_variants":[],'
                    '"variants":[{"name":"best_edit","strategy":"extractive",'
                    '"segments":[{"start":1.0,"end":20.0,"role":"hook"}]}]}'
                )
            },
            "load_duration": 0,
            "prompt_eval_count": 500,
            "eval_count": 160,
        }

        analyzer = RetentionAnalyzer(model="test-model")
        result = analyzer.analyze(
            {
                "candidate_start": 1.0,
                "candidate_end": 20.0,
                "start": 0.0,
                "end": 25.0,
                "segments": [
                    {"start": 1.0, "end": 20.0, "text": "A gameplay moment happens here."}
                ],
            },
            {"word_rate": 2.0},
        )

        self.assertEqual(mock_chat.call_count, 1)
        self.assertEqual(result["content_type"], "gaming")


if __name__ == "__main__":
    unittest.main()
