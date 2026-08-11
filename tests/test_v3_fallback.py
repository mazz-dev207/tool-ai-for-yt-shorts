import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.v3.pipeline import run_v3_editorial_stage


class V3FallbackTests(unittest.TestCase):
    def test_gemini_unavailable_keeps_v2_source_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = root / "highlights"
            transcript = root / "transcript"
            output = root / "output"
            highlights.mkdir()
            transcript.mkdir()
            video = root / "video.mp4"
            video.write_bytes(b"x")

            original = [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "duration": 10.0,
                    "hook_score": 82,
                    "hook_confidence": 0.9,
                    "segments": [
                        {"start": 10.0, "end": 14.0, "role": "escalation"},
                        {"start": 16.0, "end": 20.0, "role": "payoff"},
                    ],
                }
            ]
            (highlights / "video.json").write_text(
                json.dumps(original), encoding="utf-8"
            )
            (transcript / "video.json").write_text("[]", encoding="utf-8")

            with patch("src.v3.pipeline.HIGHLIGHTS_DIR", highlights), patch(
                "src.v3.pipeline.TRANSCRIPT_DIR", transcript
            ), patch("src.v3.pipeline.OUTPUT_DIR", output), patch(
                "src.v3.pipeline.probe_duration", return_value=30.0
            ), patch(
                "src.v3.pipeline.GeminiEditorialReasoner",
                side_effect=RuntimeError("offline"),
            ):
                run_v3_editorial_stage("video", video, "gaming")

            saved = json.loads(
                (highlights / "video.json").read_text(encoding="utf-8")
            )[0]
            self.assertEqual(
                [(x["start"], x["end"]) for x in saved["segments"]],
                [(10.0, 14.0), (16.0, 20.0)],
            )
            self.assertTrue(saved["v3"]["no_transformation_needed"])
            self.assertTrue(saved["v3"]["qa_passed"])


if __name__ == "__main__":
    unittest.main()
