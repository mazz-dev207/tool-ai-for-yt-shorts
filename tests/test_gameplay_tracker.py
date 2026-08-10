import unittest

from src.gameplay_tracker import GameplayObservation, stabilize_gameplay_observations


FRAME_W = 1920
FRAME_H = 1080
CROP_W = 608
CROP_H = 1080
CENTER_X = FRAME_W / 2
CENTER_Y = FRAME_H / 2


def obs(time, x, confidence=0.80, source="player", scene_cut=False):
    return GameplayObservation(
        time=float(time),
        center_x=float(x),
        center_y=CENTER_Y,
        confidence=float(confidence),
        source=source,
        scene_cut=scene_cut,
    )


def missing(time):
    return GameplayObservation(
        time=float(time),
        center_x=CENTER_X,
        center_y=CENTER_Y,
        confidence=0.0,
        source="no_target",
    )


def track(observations):
    return stabilize_gameplay_observations(
        observations,
        crop_width=CROP_W,
        crop_height=CROP_H,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
    )


class GameplayTrackerTests(unittest.TestCase):
    def test_static_target_keeps_crop_stable(self):
        points, _ = track([
            obs(0.0, 960), obs(0.5, 965), obs(1.0, 955), obs(1.5, 962)
        ])
        xs = [point.center_x for point in points]
        self.assertLess(max(xs) - min(xs), 5.0)

    def test_slow_right_motion_is_followed_gradually(self):
        points, _ = track([
            obs(0.0, 960), obs(0.5, 1040), obs(1.0, 1120), obs(1.5, 1200)
        ])
        self.assertGreater(points[-1].center_x, points[0].center_x)
        self.assertLess(points[-1].center_x, 1200)

    def test_small_oscillation_stays_inside_dead_zone(self):
        points, _ = track([
            obs(0.0, 960), obs(0.5, 985), obs(1.0, 935), obs(1.5, 980), obs(2.0, 940)
        ])
        xs = [point.center_x for point in points]
        self.assertLess(max(xs) - min(xs), 12.0)

    def test_opposite_ui_spike_does_not_abandon_locked_target(self):
        points, summary = track([
            obs(0.0, 820, 0.85, "player"),
            obs(0.5, 830, 0.84, "player"),
            obs(1.0, 1580, 0.55, "ui_notification"),
            obs(1.5, 835, 0.84, "player"),
        ])
        self.assertEqual(summary["target_switches"], 0)
        self.assertLess(points[2].center_x, 1100)

    def test_lost_target_half_second_holds_camera(self):
        points, summary = track([
            obs(0.0, 760), obs(0.5, 760), missing(1.0), obs(1.5, 770)
        ])
        self.assertGreaterEqual(summary["held_samples"], 1)
        self.assertLess(abs(points[2].center_x - points[1].center_x), 30.0)

    def test_lost_target_definitively_recenters_gradually(self):
        points, summary = track([
            obs(0.0, 700), obs(0.5, 700), missing(1.0), missing(1.5), missing(2.0), missing(2.5)
        ])
        self.assertGreaterEqual(summary["recenter_samples"], 1)
        # It must move toward neutral center without snapping there immediately.
        self.assertGreater(points[-1].center_x, points[1].center_x)
        self.assertLess(points[-1].center_x, CENTER_X + 1)

    def test_much_stronger_target_switches_controlled(self):
        points, summary = track([
            obs(0.0, 720, 0.45, "optical_flow_action"),
            obs(0.5, 725, 0.45, "optical_flow_action"),
            obs(1.0, 1450, 0.90, "crosshair_action"),
            obs(1.5, 1440, 0.90, "crosshair_action"),
        ])
        self.assertGreaterEqual(summary["target_switches"], 1)
        self.assertTrue(points[2].debug["TARGET_SWITCH"])
        self.assertLess(points[2].center_x, 1450)

    def test_alternating_saliency_does_not_ping_pong(self):
        points, summary = track([
            obs(0.0, 900, 0.86, "player"),
            obs(0.5, 1500, 0.50, "optical_flow_action"),
            obs(1.0, 500, 0.50, "optical_flow_action"),
            obs(1.5, 1500, 0.50, "optical_flow_action"),
            obs(2.0, 900, 0.86, "player"),
        ])
        xs = [point.center_x for point in points]
        self.assertEqual(summary["target_switches"], 0)
        self.assertLess(max(xs) - min(xs), 90.0)

    def test_scene_cut_resets_tracking_cleanly(self):
        points, summary = track([
            obs(0.0, 700, 0.85, "player"),
            obs(0.5, 710, 0.85, "player"),
            obs(1.0, 1400, 0.85, "player", scene_cut=True),
        ])
        self.assertEqual(summary["scene_cuts"], 1)
        self.assertEqual(points[-1].debug["MOVEMENT"], "SCENE_RESET")

    def test_crosshair_source_receives_priority(self):
        points, _ = track([
            obs(0.0, 760, 0.44, "optical_flow_action"),
            obs(0.5, 765, 0.44, "optical_flow_action"),
            obs(1.0, 1320, 0.78, "crosshair"),
        ])
        self.assertTrue(points[-1].debug["TARGET_SWITCH"])
        self.assertEqual(points[-1].debug["PRIMARY_TARGET"], "crosshair")

    def test_third_person_character_is_followed_with_context(self):
        points, _ = track([
            obs(0.0, 820, 0.80, "player_character"),
            obs(0.5, 900, 0.80, "player_character"),
            obs(1.0, 1000, 0.80, "player_character"),
            obs(1.5, 1120, 0.80, "player_character"),
        ])
        self.assertGreater(points[-1].center_x, points[0].center_x)
        self.assertLess(points[-1].center_x, 1120)

    def test_rapid_motion_respects_speed_limit_without_snap(self):
        points, _ = track([
            obs(0.0, 700, 0.90, "player"),
            obs(0.5, 720, 0.90, "player"),
            obs(1.0, 1500, 0.96, "crosshair_action"),
            obs(1.5, 1500, 0.96, "crosshair_action"),
        ])
        deltas = [abs(b.center_x - a.center_x) for a, b in zip(points, points[1:])]
        self.assertLess(max(deltas), CROP_W * 0.24 * 0.5 + 1.0)
        self.assertLess(points[2].center_x, 1500)


if __name__ == "__main__":
    unittest.main()
