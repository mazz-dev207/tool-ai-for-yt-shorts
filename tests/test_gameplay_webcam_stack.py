import unittest

from src.smart_crop import CropPlan
from src.smart_crop_v2 import (
    FocusPoint,
    Rect,
    SmartCropV2Plan,
    _build_layout_metadata,
    choose_stack_webcam_ratio,
    validate_layout_plan,
)
from src.smartcrop_compositor import (
    build_gameplay_webcam_stack_filter,
    build_safe_cover_stack_filter,
)


class GameplayWebcamStackTests(unittest.TestCase):
    def make_plan(self, webcam=None, webcam_ratio=0.40):
        webcam = webcam or Rect(30, 30, 420, 300)
        layout = _build_layout_metadata(webcam_ratio)
        legacy = CropPlan(1920, 1080, 608, 1080, 20.0, [])
        return SmartCropV2Plan(
            mode="GAMEPLAY_WEBCAM_STACK",
            input_width=1920,
            input_height=1080,
            duration=20.0,
            legacy_plan=legacy,
            confidence=0.90,
            webcam_region=webcam,
            gameplay_region=Rect(0, 0, 1920, 1080),
            gameplay_crop_width=900,
            gameplay_crop_height=1080,
            gameplay_output_height=layout["gameplay_height"],
            webcam_output_height=layout["webcam_height"],
            focus_points=[FocusPoint(0.0, 960.0, 540.0, "stable_gameplay_target")],
            layout_metadata=layout,
        )

    def build_filter(self, plan, blurred=True):
        return build_gameplay_webcam_stack_filter(
            plan=plan,
            subtitle_path="subs.ass",
            command_path="crop.cmd",
            initial_x=510.0,
            initial_y=0.0,
            blurred_fill=blurred,
        )

    def test_case_1_top_left_source_still_places_webcam_on_top(self):
        plan = self.make_plan(Rect(20, 20, 420, 300))
        value = self.build_filter(plan)
        self.assertIn("[cam_panel][game_panel]vstack=inputs=2", value)
        self.assertEqual(plan.layout_metadata["webcam_position"], "top")
        self.assertEqual(plan.layout_metadata["gameplay_position"], "bottom")

    def test_case_2_top_right_source_still_places_webcam_on_top(self):
        plan = self.make_plan(Rect(1480, 20, 420, 300))
        value = self.build_filter(plan)
        self.assertIn("crop=w=420:h=300:x=1480:y=20", value)
        self.assertIn("[cam_panel][game_panel]vstack=inputs=2", value)

    def test_case_3_small_webcam_uses_blurred_fill_without_black_pad(self):
        plan = self.make_plan(Rect(30, 30, 180, 130))
        value = self.build_filter(plan)
        self.assertIn("gblur=sigma=", value)
        self.assertNotIn("pad=", value)
        self.assertIn("force_original_aspect_ratio=decrease", value)

    def test_case_4_wide_webcam_uses_cover_background_and_sharp_foreground(self):
        plan = self.make_plan(Rect(50, 40, 700, 220))
        value = self.build_filter(plan)
        self.assertIn("force_original_aspect_ratio=increase", value)
        self.assertIn("force_original_aspect_ratio=decrease", value)
        self.assertIn("[cam_bg][cam_fg]overlay=", value)

    def test_case_5_gameplay_panel_has_its_own_blurred_fill(self):
        plan = self.make_plan()
        value = self.build_filter(plan)
        self.assertIn("[game_bg_src]scale=", value)
        self.assertIn("[game_bg][game_fg]overlay=", value)
        self.assertIn("gblur=sigma=", value)

    def test_case_6_no_black_bar_padding_in_stack_compositor(self):
        plan = self.make_plan()
        value = self.build_filter(plan)
        self.assertNotIn("pad=", value)
        self.assertNotIn("color=black", value)

    def test_case_7_foreground_and_background_are_separate_layers(self):
        plan = self.make_plan()
        value = self.build_filter(plan)
        self.assertIn("split=2[cam_bg_src][cam_fg_src]", value)
        self.assertIn("split=2[game_bg_src][game_fg_src]", value)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", value)

    def test_case_8_safe_fallback_has_content_cover_and_no_black_bars(self):
        plan = self.make_plan()
        value = build_safe_cover_stack_filter(
            plan=plan,
            subtitle_path="subs.ass",
            command_path="crop.cmd",
            initial_x=510.0,
            initial_y=0.0,
        )
        self.assertNotIn("gblur=", value)
        self.assertNotIn("pad=", value)
        self.assertIn("force_original_aspect_ratio=increase", value)
        self.assertIn("[cam_panel][game_panel]vstack=inputs=2", value)

    def test_case_9_layout_metadata_is_explicit_40_60(self):
        plan = self.make_plan(webcam_ratio=0.40)
        layout = plan.layout_metadata
        self.assertEqual(layout["type"], "GAMEPLAY_WEBCAM_STACK")
        self.assertEqual(layout["webcam_position"], "top")
        self.assertEqual(layout["gameplay_position"], "bottom")
        self.assertAlmostEqual(layout["webcam_ratio"], 0.40, places=2)
        self.assertAlmostEqual(layout["gameplay_ratio"], 0.60, places=2)
        self.assertEqual(layout["split_y"], plan.webcam_output_height)
        self.assertTrue(validate_layout_plan(plan))

    def test_case_10_caption_safe_area_is_below_webcam_panel(self):
        plan = self.make_plan(webcam_ratio=0.40)
        safe = plan.layout_metadata["caption_safe_area"]
        self.assertGreater(safe["y1"], plan.layout_metadata["split_y"])
        self.assertGreater(safe["x1"], 0)
        self.assertLess(safe["x2"], 1080)
        self.assertGreater(safe["y2"], safe["y1"])

    def test_reaction_ratio_is_adaptive_but_bounded(self):
        normal = choose_stack_webcam_ratio([])
        reaction = choose_stack_webcam_ratio([{"type": "reaction"}] * 2)
        strong = choose_stack_webcam_ratio([{"type": "reaction"}] * 5)
        self.assertGreater(reaction, normal)
        self.assertGreaterEqual(strong, reaction)
        self.assertGreaterEqual(normal, 0.38)
        self.assertLessEqual(strong, 0.48)


if __name__ == "__main__":
    unittest.main()
