from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from src.config import (
    ENABLE_GAMEPLAY_TARGET_TRACKING,
    GAMEPLAY_ACCELERATION_PENALTY,
    GAMEPLAY_ANTI_PINGPONG_WINDOW,
    GAMEPLAY_MAX_CROP_ACCELERATION,
    GAMEPLAY_MAX_CROP_SPEED_X,
    GAMEPLAY_MAX_CROP_SPEED_Y,
    GAMEPLAY_MOVEMENT_PENALTY,
    GAMEPLAY_PREDICTION_TIME,
    GAMEPLAY_RECENTER_SPEED,
    GAMEPLAY_SCENE_CUT_THRESHOLD,
    GAMEPLAY_TARGET_HOLD_TIME,
    GAMEPLAY_TARGET_LOCK_THRESHOLD,
    GAMEPLAY_TARGET_SWITCH_PENALTY,
    GAMEPLAY_TARGET_SWITCH_THRESHOLD,
    GAMEPLAY_TRACK_DEAD_ZONE_X,
    GAMEPLAY_TRACK_DEAD_ZONE_Y,
    GAMEPLAY_TRACK_SMOOTHING,
)


@dataclass
class GameplayObservation:
    time: float
    center_x: float
    center_y: float
    confidence: float
    source: str = "optical_flow"
    scene_cut: bool = False


@dataclass
class GameplayTrackPoint:
    time: float
    center_x: float
    center_y: float
    source: str
    debug: dict = field(default_factory=dict)


@dataclass
class GameplayTrackingState:
    current_target_x: float | None = None
    current_target_y: float | None = None
    target_id: int = 0
    target_confidence: float = 0.0
    target_locked: bool = False
    current_crop_x: float = 0.0
    current_crop_y: float = 0.0
    crop_velocity_x: float = 0.0
    crop_velocity_y: float = 0.0
    target_velocity_x: float = 0.0
    target_velocity_y: float = 0.0
    last_target_seen: float = -1e9
    last_switch_time: float = -1e9
    previous_direction_x: int = 0
    direction_changes: list[float] = field(default_factory=list)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _sign(value: float, epsilon: float = 1e-6) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _source_bonus(source: str) -> float:
    source = str(source or "").lower()
    if "crosshair" in source or "aim" in source:
        return 0.18
    if "player" in source or "character" in source or "vehicle" in source:
        return 0.15
    if "enemy" in source or "action" in source or "ball" in source or "puck" in source:
        return 0.10
    if "optical" in source:
        return 0.03
    return 0.0


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _clamp_camera_center(
    x: float,
    y: float,
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    min_x = crop_width / 2.0
    max_x = frame_width - crop_width / 2.0
    min_y = crop_height / 2.0
    max_y = frame_height - crop_height / 2.0
    return _clamp(x, min_x, max_x), _clamp(y, min_y, max_y)


def _prepare_gray(frame, max_width: int = 640):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, max_width / max(1.0, float(width)))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray, scale


def _mask_region(mask: np.ndarray, region, scale: float) -> None:
    if region is None:
        return
    x = int(max(0, getattr(region, "x", 0)) * scale)
    y = int(max(0, getattr(region, "y", 0)) * scale)
    x2 = int(max(0, getattr(region, "x2", getattr(region, "x", 0) + getattr(region, "w", 0))) * scale)
    y2 = int(max(0, getattr(region, "y2", getattr(region, "y", 0) + getattr(region, "h", 0))) * scale)
    x = max(0, min(mask.shape[1], x))
    x2 = max(0, min(mask.shape[1], x2))
    y = max(0, min(mask.shape[0], y))
    y2 = max(0, min(mask.shape[0], y2))
    if x2 > x and y2 > y:
        mask[y:y2, x:x2] = 0.0


def observe_gameplay_motion(
    previous_frame,
    current_frame,
    time_position: float,
    ignored_region=None,
) -> GameplayObservation | None:
    """Build one multi-frame gameplay observation using optical flow.

    This intentionally does not treat raw saliency as a semantic player detector.
    It produces an action-region candidate that the stateful tracker may lock,
    reject, hold, or switch away from using temporal persistence and hysteresis.
    """
    if previous_frame is None or current_frame is None:
        return None

    previous_gray, previous_scale = _prepare_gray(previous_frame)
    current_gray, current_scale = _prepare_gray(current_frame)
    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(current_gray, (previous_gray.shape[1], previous_gray.shape[0]))
    scale = min(previous_scale, current_scale)

    mean_diff = float(cv2.absdiff(previous_gray, current_gray).mean()) / 255.0
    scene_cut = mean_diff >= GAMEPLAY_SCENE_CUT_THRESHOLD

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    _mask_region(magnitude, ignored_region, scale)

    positive = magnitude[magnitude > 0.05]
    if positive.size < max(20, int(magnitude.size * 0.002)):
        return GameplayObservation(
            time=time_position,
            center_x=current_frame.shape[1] / 2.0,
            center_y=current_frame.shape[0] / 2.0,
            confidence=0.0,
            source="no_target",
            scene_cut=scene_cut,
        )

    threshold = max(0.6, float(np.percentile(positive, 72)))
    weights = np.where(magnitude >= threshold, magnitude, 0.0).astype(np.float32)

    # Mild center prior suppresses edge HUD/kill-feed flashes without forcing
    # the camera to stay centered when coherent gameplay motion is elsewhere.
    h, w = weights.shape
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - w / 2.0) / max(1.0, w / 2.0)
    ny = (yy - h / 2.0) / max(1.0, h / 2.0)
    center_prior = 1.0 - 0.28 * np.clip(np.sqrt(nx * nx + ny * ny), 0.0, 1.0)
    weights *= center_prior.astype(np.float32)

    total = float(weights.sum())
    if total <= 1e-6:
        return GameplayObservation(
            time=time_position,
            center_x=current_frame.shape[1] / 2.0,
            center_y=current_frame.shape[0] / 2.0,
            confidence=0.0,
            source="no_target",
            scene_cut=scene_cut,
        )

    center_x_small = float((weights * xx).sum() / total)
    center_y_small = float((weights * yy).sum() / total)
    inverse_scale = 1.0 / max(scale, 1e-6)
    center_x = center_x_small * inverse_scale
    center_y = center_y_small * inverse_scale

    active_ratio = float(np.count_nonzero(weights)) / max(1.0, float(weights.size))
    motion_strength = float(np.mean(positive))
    confidence = _clamp(
        0.18 + active_ratio * 4.0 + min(0.48, motion_strength / 18.0),
        0.0,
        0.96,
    )

    return GameplayObservation(
        time=time_position,
        center_x=center_x,
        center_y=center_y,
        confidence=confidence,
        source="optical_flow_action",
        scene_cut=scene_cut,
    )


def build_gameplay_observations(samples, ignored_region=None) -> list[GameplayObservation]:
    observations: list[GameplayObservation] = []
    previous_frame = None
    for time_position, frame, *_rest in samples:
        observation = observe_gameplay_motion(
            previous_frame,
            frame,
            float(time_position),
            ignored_region=ignored_region,
        )
        if observation is None:
            observation = GameplayObservation(
                time=float(time_position),
                center_x=frame.shape[1] / 2.0,
                center_y=frame.shape[0] / 2.0,
                confidence=0.0,
                source="initial_center",
                scene_cut=False,
            )
        observations.append(observation)
        previous_frame = frame
    return observations


def stabilize_gameplay_observations(
    observations: list[GameplayObservation],
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[list[GameplayTrackPoint], dict]:
    """Convert noisy action observations into a stable virtual camera path."""
    if not observations:
        cx, cy = _clamp_camera_center(
            frame_width / 2.0,
            frame_height / 2.0,
            crop_width,
            crop_height,
            frame_width,
            frame_height,
        )
        return [GameplayTrackPoint(0.0, cx, cy, "neutral")], {
            "target_switches": 0,
            "direction_changes": 0,
            "scene_cuts": 0,
            "locked_samples": 0,
            "held_samples": 0,
            "recenter_samples": 0,
        }

    center_x, center_y = _clamp_camera_center(
        frame_width / 2.0,
        frame_height / 2.0,
        crop_width,
        crop_height,
        frame_width,
        frame_height,
    )
    state = GameplayTrackingState(current_crop_x=center_x, current_crop_y=center_y)
    result: list[GameplayTrackPoint] = []
    summary = {
        "target_switches": 0,
        "direction_changes": 0,
        "scene_cuts": 0,
        "locked_samples": 0,
        "held_samples": 0,
        "recenter_samples": 0,
    }
    previous_time = float(observations[0].time)

    for observation in observations:
        now = float(observation.time)
        dt = max(0.001, now - previous_time) if result else 0.001
        previous_time = now
        target_switch = False
        movement_mode = "HOLD"

        if observation.scene_cut:
            summary["scene_cuts"] += 1
            state.target_locked = False
            state.target_confidence = 0.0
            state.current_target_x = None
            state.current_target_y = None
            state.crop_velocity_x = 0.0
            state.crop_velocity_y = 0.0
            state.target_velocity_x = 0.0
            state.target_velocity_y = 0.0
            state.direction_changes.clear()

        valid_observation = observation.confidence > 0.0
        candidate_score = _clamp(observation.confidence + _source_bonus(observation.source), 0.0, 1.0)

        if valid_observation:
            if state.current_target_x is None or state.current_target_y is None:
                if candidate_score >= GAMEPLAY_TARGET_LOCK_THRESHOLD or observation.scene_cut:
                    state.target_id += 1
                    state.current_target_x = observation.center_x
                    state.current_target_y = observation.center_y
                    state.target_confidence = candidate_score
                    state.target_locked = True
                    state.last_target_seen = now
                    state.last_switch_time = now
                    target_switch = True
            else:
                target_distance = _distance(
                    state.current_target_x,
                    state.current_target_y,
                    observation.center_x,
                    observation.center_y,
                )
                same_target_radius = max(crop_width * 0.42, crop_height * 0.32)
                persistence_bonus = 0.12 if target_distance <= same_target_radius else 0.0
                candidate_score = _clamp(candidate_score + persistence_bonus, 0.0, 1.0)

                if target_distance <= same_target_radius:
                    old_x = state.current_target_x
                    old_y = state.current_target_y
                    blend = 0.62
                    state.current_target_x = old_x + (observation.center_x - old_x) * blend
                    state.current_target_y = old_y + (observation.center_y - old_y) * blend
                    state.target_velocity_x = (state.current_target_x - old_x) / dt
                    state.target_velocity_y = (state.current_target_y - old_y) / dt
                    state.target_confidence = max(candidate_score, state.target_confidence * 0.92)
                    state.target_locked = True
                    state.last_target_seen = now
                else:
                    switch_required = (
                        state.target_confidence
                        + GAMEPLAY_TARGET_SWITCH_THRESHOLD
                        + GAMEPLAY_TARGET_SWITCH_PENALTY
                    )
                    target_missing = now - state.last_target_seen > GAMEPLAY_TARGET_HOLD_TIME
                    if candidate_score >= switch_required or target_missing or observation.scene_cut:
                        old_x = state.current_target_x
                        old_y = state.current_target_y
                        state.target_id += 1
                        state.current_target_x = observation.center_x
                        state.current_target_y = observation.center_y
                        state.target_velocity_x = (observation.center_x - old_x) / dt
                        state.target_velocity_y = (observation.center_y - old_y) / dt
                        state.target_confidence = candidate_score
                        state.target_locked = True
                        state.last_target_seen = now
                        state.last_switch_time = now
                        target_switch = True
                        summary["target_switches"] += 1
                    else:
                        # Reject one-frame saliency/action spikes on the opposite side.
                        state.target_confidence *= 0.97
        elif state.target_locked:
            state.target_confidence *= 0.94

        time_since_seen = now - state.last_target_seen
        holding_lost_target = (
            state.current_target_x is not None
            and time_since_seen <= GAMEPLAY_TARGET_HOLD_TIME
        )
        if holding_lost_target and not valid_observation:
            summary["held_samples"] += 1

        if state.current_target_x is not None and (
            state.target_locked or holding_lost_target
        ):
            if state.target_locked:
                summary["locked_samples"] += 1
            target_x = state.current_target_x
            target_y = state.current_target_y
            predicted_x = target_x + state.target_velocity_x * GAMEPLAY_PREDICTION_TIME
            predicted_y = target_y + state.target_velocity_y * GAMEPLAY_PREDICTION_TIME

            dead_x = crop_width * GAMEPLAY_TRACK_DEAD_ZONE_X
            dead_y = crop_height * GAMEPLAY_TRACK_DEAD_ZONE_Y
            offset_x = target_x - state.current_crop_x
            offset_y = target_y - state.current_crop_y

            desired_x = state.current_crop_x
            desired_y = state.current_crop_y
            if abs(offset_x) > dead_x:
                desired_x = predicted_x - math.copysign(dead_x, offset_x)
                movement_mode = "FOLLOW_RIGHT" if offset_x > 0 else "FOLLOW_LEFT"
            if abs(offset_y) > dead_y:
                desired_y = predicted_y - math.copysign(dead_y, offset_y)
                if movement_mode == "HOLD":
                    movement_mode = "FOLLOW_DOWN" if offset_y > 0 else "FOLLOW_UP"
        else:
            state.target_locked = False
            desired_x = frame_width / 2.0
            desired_y = frame_height / 2.0
            movement_mode = "RECENTER"
            summary["recenter_samples"] += 1

        desired_x, desired_y = _clamp_camera_center(
            desired_x,
            desired_y,
            crop_width,
            crop_height,
            frame_width,
            frame_height,
        )

        if observation.scene_cut:
            # A hard cut may hide a camera reposition; reset instead of animating
            # across unrelated scenes.
            state.current_crop_x = desired_x
            state.current_crop_y = desired_y
            state.crop_velocity_x = 0.0
            state.crop_velocity_y = 0.0
            movement_mode = "SCENE_RESET"
        elif ENABLE_GAMEPLAY_TARGET_TRACKING:
            smoothing = GAMEPLAY_TRACK_SMOOTHING
            if movement_mode == "RECENTER":
                smoothing = min(smoothing, GAMEPLAY_RECENTER_SPEED)

            delta_x = desired_x - state.current_crop_x
            delta_y = desired_y - state.current_crop_y
            distance_norm = math.hypot(
                delta_x / max(1.0, crop_width),
                delta_y / max(1.0, crop_height),
            )
            movement_factor = 1.0 / (1.0 + GAMEPLAY_MOVEMENT_PENALTY * distance_norm)
            requested_vx = (delta_x * smoothing * movement_factor) / dt
            requested_vy = (delta_y * smoothing * movement_factor) / dt

            max_vx = crop_width * GAMEPLAY_MAX_CROP_SPEED_X
            max_vy = crop_height * GAMEPLAY_MAX_CROP_SPEED_Y
            requested_vx = _clamp(requested_vx, -max_vx, max_vx)
            requested_vy = _clamp(requested_vy, -max_vy, max_vy)

            max_ax = crop_width * GAMEPLAY_MAX_CROP_ACCELERATION
            max_ay = crop_height * GAMEPLAY_MAX_CROP_ACCELERATION
            accel_factor = max(0.20, 1.0 - GAMEPLAY_ACCELERATION_PENALTY)
            dvx = _clamp(
                requested_vx - state.crop_velocity_x,
                -max_ax * dt * accel_factor,
                max_ax * dt * accel_factor,
            )
            dvy = _clamp(
                requested_vy - state.crop_velocity_y,
                -max_ay * dt * accel_factor,
                max_ay * dt * accel_factor,
            )
            new_vx = state.crop_velocity_x + dvx
            new_vy = state.crop_velocity_y + dvy

            direction_x = _sign(new_vx, max_vx * 0.03)
            if (
                direction_x
                and state.previous_direction_x
                and direction_x != state.previous_direction_x
            ):
                state.direction_changes.append(now)
                summary["direction_changes"] += 1
            if direction_x:
                state.previous_direction_x = direction_x
            state.direction_changes = [
                timestamp
                for timestamp in state.direction_changes
                if now - timestamp <= GAMEPLAY_ANTI_PINGPONG_WINDOW
            ]
            if len(state.direction_changes) >= 2:
                new_vx *= 0.28
                new_vy *= 0.55
                movement_mode = "ANTI_PINGPONG"

            state.crop_velocity_x = new_vx
            state.crop_velocity_y = new_vy
            state.current_crop_x += new_vx * dt
            state.current_crop_y += new_vy * dt
            state.current_crop_x, state.current_crop_y = _clamp_camera_center(
                state.current_crop_x,
                state.current_crop_y,
                crop_width,
                crop_height,
                frame_width,
                frame_height,
            )
        else:
            state.current_crop_x = desired_x
            state.current_crop_y = desired_y

        debug = {
            "PRIMARY_TARGET": observation.source if valid_observation else "none",
            "TARGET_ID": state.target_id,
            "TARGET_CONFIDENCE": round(state.target_confidence, 4),
            "TARGET_CENTER": (
                [round(state.current_target_x, 2), round(state.current_target_y, 2)]
                if state.current_target_x is not None and state.current_target_y is not None
                else None
            ),
            "DESIRED_CROP_CENTER": [round(desired_x, 2), round(desired_y, 2)],
            "ACTUAL_CROP_CENTER": [round(state.current_crop_x, 2), round(state.current_crop_y, 2)],
            "CROP_VELOCITY": [round(state.crop_velocity_x, 2), round(state.crop_velocity_y, 2)],
            "DEAD_ZONE": [
                round(crop_width * GAMEPLAY_TRACK_DEAD_ZONE_X, 2),
                round(crop_height * GAMEPLAY_TRACK_DEAD_ZONE_Y, 2),
            ],
            "TARGET_LOCK": bool(state.target_locked),
            "TARGET_SWITCH": bool(target_switch),
            "MOVEMENT": movement_mode,
            "SCENE_CUT": bool(observation.scene_cut),
        }
        result.append(
            GameplayTrackPoint(
                time=now,
                center_x=state.current_crop_x,
                center_y=state.current_crop_y,
                source="stable_gameplay_target" if state.target_locked else "stable_gameplay_hold",
                debug=debug,
            )
        )

    return result, summary


def build_stable_gameplay_track(
    samples,
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
    ignored_region=None,
) -> tuple[list[GameplayTrackPoint], dict]:
    observations = build_gameplay_observations(samples, ignored_region=ignored_region)
    return stabilize_gameplay_observations(
        observations,
        crop_width,
        crop_height,
        frame_width,
        frame_height,
    )
