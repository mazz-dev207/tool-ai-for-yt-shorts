from __future__ import annotations

import os
from pathlib import Path

from src.gameplay_tracker import build_stable_gameplay_track
from src.logger import info, warning
from src.reading_focus import apply_reading_focus
from src.smart_crop_v2 import (
    FocusPoint,
    Rect,
    SmartCropV2Plan,
    _build_layout_metadata,
    _calculate_crop_size_for_aspect,
    _detect_reaction_events,
    _sample_video,
    validate_crop_bounds,
)
from src.webcam_detector import build_webcam_candidates


FORCED_WEBCAM_RATIO = float(os.getenv("FORCED_WEBCAM_RATIO", "0.38"))
FORCED_WEBCAM_RATIO_MIN = float(os.getenv("FORCED_WEBCAM_RATIO_MIN", "0.35"))
FORCED_WEBCAM_RATIO_MAX = float(os.getenv("FORCED_WEBCAM_RATIO_MAX", "0.40"))
FORCED_REACTION_WEBCAM_RATIO = float(os.getenv("FORCED_REACTION_WEBCAM_RATIO", "0.40"))
FORCED_REACTION_EVENT_THRESHOLD = max(
    1,
    int(os.getenv("FORCED_REACTION_EVENT_THRESHOLD", "2")),
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _forced_ratio(reaction_events: list[dict]) -> float:
    if len(reaction_events or []) >= FORCED_REACTION_EVENT_THRESHOLD:
        ratio = FORCED_REACTION_WEBCAM_RATIO
    else:
        ratio = FORCED_WEBCAM_RATIO
    return _clamp(ratio, FORCED_WEBCAM_RATIO_MIN, FORCED_WEBCAM_RATIO_MAX)


def build_forced_gameplay_webcam_plan(
    video_path: Path,
    plan: SmartCropV2Plan,
) -> SmartCropV2Plan:
    """Build GAMEPLAY_WEBCAM_STACK when the caller explicitly knows a webcam exists.

    In auto mode, the webcam detector decides both IF and WHERE a webcam exists.
    In forced mode, the user supplies the IF prior; the detector only ranks WHERE
    the most plausible webcam region is. This intentionally avoids falling back to
    GAMEPLAY_ONLY just because the generic webcam threshold is conservative.
    """
    width, height, duration, samples = _sample_video(Path(video_path))
    if not samples:
        raise RuntimeError("Forced gameplay-webcam mode has no sampled frames")
    if width <= height:
        warning(
            "[SMARTCROP] Forced gameplay-webcam source is not landscape; "
            "continui, dar layout-ul poate fi suboptimal."
        )

    candidates = build_webcam_candidates(samples, width, height)
    if not candidates:
        raise RuntimeError(
            "Forced gameplay-webcam mode: nu am găsit nicio regiune webcam plauzibilă."
        )

    selected = candidates[0]
    webcam = validate_crop_bounds(
        Rect(selected.roi.x, selected.roi.y, selected.roi.w, selected.roi.h),
        width,
        height,
    )

    if selected.final_score < 0.35:
        warning(
            "[WebcamDetector] Forced mode selected a weak candidate | "
            f"score={selected.final_score:.2f} | persistence={selected.face_persistence_score:.2f} | "
            f"stability={selected.roi_stability_score:.2f}"
        )

    reaction_events = _detect_reaction_events(samples, webcam)
    webcam_ratio = _forced_ratio(reaction_events)
    layout = _build_layout_metadata(webcam_ratio)
    # Forced mode has a stricter composition contract than auto reaction mode.
    layout["webcam_ratio"] = round(webcam_ratio, 4)
    layout["gameplay_ratio"] = round(1.0 - webcam_ratio, 4)
    layout["webcam_height"] = int(round(1920 * webcam_ratio))
    if layout["webcam_height"] % 2:
        layout["webcam_height"] -= 1
    layout["gameplay_height"] = 1920 - layout["webcam_height"]
    layout["split_y"] = layout["webcam_height"]
    layout["forced_mode"] = True
    layout["ratio_contract"] = {
        "webcam_min": FORCED_WEBCAM_RATIO_MIN,
        "webcam_max": FORCED_WEBCAM_RATIO_MAX,
        "gameplay_min": round(1.0 - FORCED_WEBCAM_RATIO_MAX, 4),
        "gameplay_max": round(1.0 - FORCED_WEBCAM_RATIO_MIN, 4),
    }

    webcam_output_height = int(layout["webcam_height"])
    gameplay_output_height = int(layout["gameplay_height"])
    gameplay_target_ratio = 1080 / float(gameplay_output_height)
    crop_w, crop_h = _calculate_crop_size_for_aspect(
        width,
        height,
        gameplay_target_ratio,
    )

    track, tracking_summary = build_stable_gameplay_track(
        samples,
        crop_width=crop_w,
        crop_height=crop_h,
        frame_width=width,
        frame_height=height,
        ignored_region=webcam,
    )

    plan.mode = "GAMEPLAY_WEBCAM_STACK"
    plan.input_width = width
    plan.input_height = height
    plan.duration = duration
    plan.confidence = max(float(plan.confidence), float(selected.final_score))
    plan.webcam_region = webcam
    plan.gameplay_region = Rect(0, 0, width, height)
    plan.gameplay_crop_width = crop_w
    plan.gameplay_crop_height = crop_h
    plan.gameplay_output_height = gameplay_output_height
    plan.webcam_output_height = webcam_output_height
    plan.focus_points = [
        FocusPoint(item.time, item.center_x, item.center_y, item.source)
        for item in track
    ]
    plan.reaction_events = reaction_events
    plan.tracking_debug = [item.debug for item in track]
    plan.tracking_summary = tracking_summary
    plan.webcam_candidates = [candidate.to_debug_dict() for candidate in candidates]
    plan.webcam_detection_debug = {
        **selected.to_debug_dict(),
        "selected": True,
        "forced": True,
        "reason": "user_declared_gameplay_webcam_choose_best_candidate",
    }
    plan.layout_metadata = layout
    plan.decision_reason = (
        "forced gameplay-webcam mode + best ranked webcam candidate + "
        "35-40% top webcam / 60-65% bottom gameplay"
    )

    info(
        "[SmartCropMode] forced=gameplay-webcam | "
        f"candidate_score={selected.final_score:.2f} | source_position={selected.corner}"
    )
    info(
        "[LayoutPolicy] forced contract | webcam_position=top | gameplay_position=bottom | "
        f"webcam_ratio={webcam_ratio:.2f} | gameplay_ratio={1.0 - webcam_ratio:.2f} | "
        f"split_y={webcam_output_height}"
    )
    return plan


def enhance_plan_with_reading_focus(
    video_path: Path,
    plan: SmartCropV2Plan,
) -> SmartCropV2Plan:
    """Apply persistent text/UI reading targets on top of the stable gameplay path."""
    if plan.mode not in {"GAMEPLAY_ONLY", "GAMEPLAY_WEBCAM", "GAMEPLAY_WEBCAM_STACK"}:
        return plan
    if not plan.focus_points or plan.gameplay_crop_width <= 0 or plan.gameplay_crop_height <= 0:
        return plan

    try:
        width, height, _duration, samples = _sample_video(Path(video_path))
        ignored_region = plan.webcam_region if plan.mode in {"GAMEPLAY_WEBCAM", "GAMEPLAY_WEBCAM_STACK"} else None
        adjusted, reading_summary = apply_reading_focus(
            plan.focus_points,
            samples,
            crop_width=plan.gameplay_crop_width,
            crop_height=plan.gameplay_crop_height,
            frame_width=width,
            frame_height=height,
            ignored_region=ignored_region,
        )
        if not adjusted:
            return plan

        plan.focus_points = [
            FocusPoint(item.time, item.center_x, item.center_y, item.source)
            for item in adjusted
        ]

        existing_debug = list(plan.tracking_debug or [])
        merged_debug = []
        for index, item in enumerate(adjusted):
            base = dict(existing_debug[index]) if index < len(existing_debug) else {}
            base.update(item.debug)
            merged_debug.append(base)
        plan.tracking_debug = merged_debug
        plan.tracking_summary = {
            **(plan.tracking_summary or {}),
            **reading_summary,
        }

        active = int(reading_summary.get("reading_focus_samples", 0))
        events = int(reading_summary.get("reading_focus_events", 0))
        max_confidence = float(reading_summary.get("reading_focus_max_confidence", 0.0))
        info(
            "[ReadingFocus] "
            f"events={events} | active_samples={active} | max_confidence={max_confidence:.2f}"
        )
        if active > 0:
            plan.decision_reason = f"{plan.decision_reason} + reading-focus semantic tracking"
        return plan
    except Exception as exc:
        warning(f"[ReadingFocus] analysis failed: {exc}; păstrez gameplay tracker-ul existent.")
        return plan
