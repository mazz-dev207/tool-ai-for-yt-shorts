import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.strategy.format_intelligence import apply_format_intelligence
from src.strategy.library import (
    load_channel_strategy,
    load_format_library,
    resolve_format_library_path,
    resolve_strategy_path,
)


def entertainment_candidate():
    return {
        "start": 10.0,
        "end": 26.0,
        "title": "Normal conversation suddenly goes wrong",
        "hook": "We were just talking when something unexpected happened",
        "payoff": "Everyone screamed and laughed. No way.",
        "reason": "A normal creator interaction turns into an unexpected reveal and reaction.",
        "scores": {
            "hook": 88,
            "payoff": 91,
            "standalone_context": 86,
            "retention": 89,
            "emotional_intensity": 90,
            "entertainment": 94,
            "viral_potential": 87,
            "pacing": 82,
        },
    }


class EntertainmentStrategyV3Tests(unittest.TestCase):
    def test_entertainment_profile_resolves_dedicated_strategy_and_library(self):
        strategy_path = resolve_strategy_path("entertainment", None)
        library_path = resolve_format_library_path("entertainment")

        self.assertIsNotNone(strategy_path)
        self.assertIsNotNone(library_path)
        self.assertEqual(strategy_path.name, "channel_strategy_mazclips_entertainment.json")
        self.assertEqual(library_path.name, "mazclips_entertainment_formats.json")

    def test_entertainment_library_contains_only_entertainment_formats(self):
        strategy_path = resolve_strategy_path("entertainment", None)
        library_path = resolve_format_library_path("entertainment")
        strategy = load_channel_strategy(strategy_path)
        formats = load_format_library(library_path)

        self.assertEqual(strategy.primary_niche, "entertainment")
        self.assertEqual(strategy.channel_name, "MazClips Entertainment")
        self.assertGreaterEqual(len(formats), 6)
        self.assertTrue(all(item.niche == "entertainment" for item in formats))
        self.assertTrue(all(item.id.startswith("entertainment_") for item in formats))
        self.assertFalse(any(item.id.startswith("gaming_") for item in formats))

    def test_gaming_and_entertainment_libraries_are_isolated(self):
        gaming = load_format_library(resolve_format_library_path("gaming"))
        entertainment = load_format_library(resolve_format_library_path("entertainment"))

        gaming_ids = {item.id for item in gaming}
        entertainment_ids = {item.id for item in entertainment}

        self.assertTrue(all(item.niche == "gaming" for item in gaming))
        self.assertTrue(all(item.niche == "entertainment" for item in entertainment))
        self.assertTrue(gaming_ids.isdisjoint(entertainment_ids))

    def test_apply_format_intelligence_entertainment_never_uses_gaming_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = root / "highlights"
            transcript_dir = root / "transcript"
            highlights.mkdir()
            transcript_dir.mkdir()

            (highlights / "video.json").write_text(
                json.dumps([entertainment_candidate()]),
                encoding="utf-8",
            )
            (transcript_dir / "video.json").write_text(
                json.dumps([
                    {
                        "start": 10,
                        "end": 26,
                        "text": "We were just talking. Suddenly, what is that? No way! Everyone laughed and screamed.",
                    }
                ]),
                encoding="utf-8",
            )

            with (
                patch("src.strategy.format_intelligence.HIGHLIGHTS_DIR", highlights),
                patch("src.strategy.format_intelligence.TRANSCRIPT_DIR", transcript_dir),
            ):
                apply_format_intelligence(
                    "video",
                    "entertainment",
                    skip_low_fit=True,
                )

            saved = json.loads((highlights / "video.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved), 1)
            self.assertTrue(saved[0]["format_id"].startswith("entertainment_"))
            self.assertFalse(saved[0]["format_id"].startswith("gaming_"))
            self.assertEqual(saved[0]["strategic_gate"], "PASS")

            metadata = json.loads(
                (highlights / "video_format_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["channel_strategy"]["primary_niche"], "entertainment")
            self.assertEqual(metadata["output_candidates"], 1)
            self.assertEqual(metadata["skipped_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
