import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import chunk_transcript as chunk_module
from src import transcribe as transcribe_module
from src.highlights import candidate_generator as candidate_module


class FastPipelineCacheV3Tests(unittest.TestCase):
    def test_whisper_cache_can_restore_after_normal_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            output = root / "transcript" / "video.json"
            video = root / "video.mp4"
            video.write_bytes(b"fake-video")

            with patch.object(transcribe_module, "_TRANSCRIPT_CACHE_DIR", cache), patch.object(
                transcribe_module, "_FAST_CACHE_ENABLED", True
            ):
                cached = transcribe_module._cache_path(video)
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(
                    json.dumps(
                        [
                            {
                                "start": 0.0,
                                "end": 1.0,
                                "text": "hello",
                                "words": [],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                self.assertTrue(transcribe_module._restore_cache(video, output))

            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["text"], "hello")

    def test_chunk_cache_restores_identical_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_dir = root / "transcript"
            temp_dir = root / "temp"
            cache_dir = root / "cache"
            transcript_dir.mkdir()
            transcript = [
                {
                    "start": 0.0,
                    "end": 20.0,
                    "text": "hello world",
                    "words": [],
                }
            ]
            (transcript_dir / "video.json").write_text(
                json.dumps(transcript), encoding="utf-8"
            )

            with patch.object(chunk_module, "TRANSCRIPT_DIR", transcript_dir), patch.object(
                chunk_module, "TEMP_DIR", temp_dir
            ), patch.object(chunk_module, "_CHUNK_CACHE_DIR", cache_dir), patch.object(
                chunk_module, "_FAST_CACHE_ENABLED", True
            ):
                first = chunk_module.chunk_transcript("video")
                expected = first.read_text(encoding="utf-8")
                first.unlink()
                second = chunk_module.chunk_transcript("video")

            self.assertEqual(second.read_text(encoding="utf-8"), expected)

    def test_candidate_discovery_cache_skips_second_qwen_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            temp_dir = root / "temp"
            highlights_dir = root / "highlights"
            cache_dir = root / "cache"
            temp_dir.mkdir()
            highlights_dir.mkdir()
            (temp_dir / "video_chunks.json").write_text(
                json.dumps(
                    [
                        {
                            "start": 0.0,
                            "end": 60.0,
                            "segments": [
                                {"start": 0.0, "end": 10.0, "text": "test", "words": []}
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            calls = {"count": 0}

            def fake_select(video_name, content_profile="auto", recovery_enabled=False):
                calls["count"] += 1
                output = highlights_dir / f"{video_name}.json"
                output.write_text(
                    json.dumps(
                        [
                            {
                                "start": 1.0,
                                "end": 16.0,
                                "title": "cached candidate",
                                "scores": {"viral_potential": 90},
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                (highlights_dir / f"{video_name}_analysis.json").write_text(
                    json.dumps({"windows_analyzed": 1}), encoding="utf-8"
                )
                return output

            with patch.object(candidate_module, "TEMP_DIR", temp_dir), patch.object(
                candidate_module, "HIGHLIGHTS_DIR", highlights_dir
            ), patch.object(candidate_module, "_DISCOVERY_CACHE_DIR", cache_dir), patch.object(
                candidate_module, "_FAST_CACHE_ENABLED", True
            ), patch.object(
                candidate_module.legacy_selector, "load_selector_prompt", return_value="prompt-v1"
            ), patch.object(
                candidate_module.legacy_selector, "select_highlights", side_effect=fake_select
            ):
                first = candidate_module.generate_candidates(
                    "video", high_recall=False, content_profile="gaming"
                )
                self.assertTrue(first.exists())
                first.unlink()
                analysis = highlights_dir / "video_analysis.json"
                if analysis.exists():
                    analysis.unlink()

                second = candidate_module.generate_candidates(
                    "video", high_recall=False, content_profile="gaming"
                )

            self.assertEqual(calls["count"], 1)
            self.assertTrue(second.exists())
            self.assertEqual(
                json.loads(second.read_text(encoding="utf-8"))[0]["title"],
                "cached candidate",
            )

    def test_fast_cache_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_dir = root / "transcript"
            temp_dir = root / "temp"
            cache_dir = root / "cache"
            transcript_dir.mkdir()
            (transcript_dir / "video.json").write_text(
                json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "words": []}]),
                encoding="utf-8",
            )
            with patch.object(chunk_module, "TRANSCRIPT_DIR", transcript_dir), patch.object(
                chunk_module, "TEMP_DIR", temp_dir
            ), patch.object(chunk_module, "_CHUNK_CACHE_DIR", cache_dir), patch.object(
                chunk_module, "_FAST_CACHE_ENABLED", False
            ):
                chunk_module.chunk_transcript("video")
            self.assertFalse(cache_dir.exists())


if __name__ == "__main__":
    unittest.main()
