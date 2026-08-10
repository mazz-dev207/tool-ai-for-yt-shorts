import unittest
from pathlib import Path
from unittest.mock import patch

from src.renderer import _render_gameplay_webcam_stack
from src.smart_crop import CropPlan
from src.smart_crop_v2 import FocusPoint, Rect, SmartCropV2Plan, _build_layout_metadata


class RendererWebcamStackTests(unittest.TestCase):
    def make_plan(self):
        layout = _build_layout_metadata(0.40)
        return SmartCropV2Plan(
            mode="GAMEPLAY_WEBCAM_STACK",
            input_width=1920,
            input_height=1080,
            duration=10.0,
            legacy_plan=CropPlan(1920, 1080, 608, 1080, 10.0, []),
            webcam_region=Rect(1450, 30, 420, 300),
            gameplay_region=Rect(0, 0, 1920, 1080),
            gameplay_crop_width=900,
            gameplay_crop_height=1080,
            gameplay_output_height=layout["gameplay_height"],
            webcam_output_height=layout["webcam_height"],
            focus_points=[FocusPoint(0.0, 960.0, 540.0, "stable_gameplay_target")],
            layout_metadata=layout,
        )

    @patch("src.renderer.write_gameplay_sendcmd")
    @patch("src.renderer.build_gameplay_webcam_stack_filter", return_value="blur-filter")
    @patch("src.renderer.build_safe_cover_stack_filter", return_value="safe-cover-filter")
    @patch("src.renderer._run_stack_filter")
    def test_blur_failure_retries_safe_cover_without_crash(
        self,
        run_filter,
        safe_builder,
        blur_builder,
        write_sendcmd,
    ):
        run_filter.side_effect = [RuntimeError("blur failed"), None]
        _render_gameplay_webcam_stack(
            Path("clip.mp4"),
            Path("clip.ass"),
            Path("out.mp4"),
            self.make_plan(),
            "clip_1",
        )
        self.assertEqual(run_filter.call_count, 2)
        blur_builder.assert_called_once()
        safe_builder.assert_called_once()
        self.assertEqual(run_filter.call_args_list[1].args[2], "safe-cover-filter")
        write_sendcmd.assert_called_once()


if __name__ == "__main__":
    unittest.main()
