import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from main import parse_args


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

    def test_help_formats_without_percent_error(self):
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                parse_args()

        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("35-40 la sută", rendered)
        self.assertIn("60-65 la sută", rendered)


if __name__ == "__main__":
    unittest.main()
