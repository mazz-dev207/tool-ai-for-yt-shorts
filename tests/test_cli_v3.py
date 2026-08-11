import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from main import (
    _cleanup_targets,
    _prepare_v3_resume_checkpoint,
    _save_pre_v3_checkpoint,
    _validate_resume_prerequisites,
    parse_args,
)


class V3CliTests(unittest.TestCase):
    def test_valid_v3_cli_arguments_parse(self):
        argv = [
            "main.py",
            "video1",
            "--highlight-mode",
            "legacy",
            "--content-profile",
            "gaming",
            "--smartcrop-mode",
            "gameplay-webcam",
            "--v3-mode",
            "on",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.video_name, "video1")
        self.assertEqual(args.highlight_mode, "legacy")
        self.assertEqual(args.content_profile, "gaming")
        self.assertEqual(args.smartcrop_mode, "gameplay-webcam")
        self.assertEqual(args.v3_mode, "on")
        self.assertIsNone(args.resume_from)

    def test_resume_from_v3_parses(self):
        argv = [
            "main.py",
            "video1",
            "--resume-from",
            "v3",
            "--content-profile",
            "gaming",
            "--smartcrop-mode",
            "gameplay-only",
            "--v3-mode",
            "on",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.resume_from, "v3")
        self.assertEqual(args.v3_mode, "on")
        self.assertEqual(args.smartcrop_mode, "gameplay-only")

    def test_help_formats_without_percent_error(self):
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                parse_args()

        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("35-40 la sută", rendered)
        self.assertIn("60-65 la sută", rendered)
        self.assertIn("--resume-from {v3}", rendered)

    def test_resume_requires_existing_transcript_and_highlights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_dir = root / "transcript"
            highlights_dir = root / "highlights"
            transcript_dir.mkdir()
            highlights_dir.mkdir()

            with (
                patch("main.TRANSCRIPT_DIR", transcript_dir),
                patch("main.HIGHLIGHTS_DIR", highlights_dir),
            ):
                with self.assertRaises(FileNotFoundError):
                    _validate_resume_prerequisites("video1", "v3")

                (transcript_dir / "video1.json").write_text("[]", encoding="utf-8")
                (highlights_dir / "video1.json").write_text("[]", encoding="utf-8")
                _validate_resume_prerequisites("video1", "v3")

    def test_resume_cleanup_preserves_upstream_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_dir = root / "transcript"
            highlights_dir = root / "highlights"
            output_dir = root / "output"
            subtitles_dir = root / "subtitles"
            final_dir = root / "final"
            temp_dir = root / "temp"

            with (
                patch("main.TRANSCRIPT_DIR", transcript_dir),
                patch("main.HIGHLIGHTS_DIR", highlights_dir),
                patch("main.OUTPUT_DIR", output_dir),
                patch("main.SUBTITLES_DIR", subtitles_dir),
                patch("main.FINAL_DIR", final_dir),
                patch("main.TEMP_DIR", temp_dir),
            ):
                targets = _cleanup_targets("video1", "v3")

            self.assertNotIn(transcript_dir, targets)
            self.assertNotIn(highlights_dir, targets)
            self.assertNotIn(temp_dir, targets)
            self.assertIn(output_dir, targets)
            self.assertIn(subtitles_dir, targets)
            self.assertIn(final_dir, targets)
            self.assertIn(highlights_dir / "v3" / "video1", targets)

    def test_save_pre_v3_checkpoint_copies_working_highlights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights_dir = root / "highlights"
            highlights_dir.mkdir()
            working = highlights_dir / "video1.json"
            working.write_text('[{"start": 10, "end": 20}]', encoding="utf-8")

            with patch("main.HIGHLIGHTS_DIR", highlights_dir):
                checkpoint = _save_pre_v3_checkpoint("video1")

            self.assertTrue(checkpoint.exists())
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), working.read_text(encoding="utf-8"))

    def test_resume_checkpoint_restores_baseline_without_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights_dir = root / "highlights"
            checkpoint_dir = highlights_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            working = highlights_dir / "video1.json"
            checkpoint = checkpoint_dir / "video1_pre_v3.json"
            checkpoint.write_text('[{"start": 10, "end": 20, "segments": [{"start": 10, "end": 12}]}]', encoding="utf-8")
            working.write_text('[{"start": 9, "end": 25, "segments": [{"start": 9, "end": 25}]}]', encoding="utf-8")

            with patch("main.HIGHLIGHTS_DIR", highlights_dir):
                restored = _prepare_v3_resume_checkpoint("video1")

            self.assertEqual(restored, checkpoint)
            self.assertEqual(working.read_text(encoding="utf-8"), checkpoint.read_text(encoding="utf-8"))

    def test_first_resume_creates_baseline_when_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights_dir = root / "highlights"
            highlights_dir.mkdir()
            working = highlights_dir / "video1.json"
            working.write_text('[{"start": 30, "end": 40}]', encoding="utf-8")

            with patch("main.HIGHLIGHTS_DIR", highlights_dir):
                checkpoint = _prepare_v3_resume_checkpoint("video1")

            self.assertTrue(checkpoint.exists())
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), working.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
