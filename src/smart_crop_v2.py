from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from src.config import (
    CAPTION_MARGIN_L,
    CAPTION_MARGIN_R,
    HIGHLIGHTS_DIR,
    SMARTCROP_BLURRED_FILL_ENABLED,
    SMARTCROP_DEBUG,
    SMARTCROP_DETECT_GAMEPLAY_WEBCAM,
    SMARTCROP_MAX_CROP_VELOCITY,
    SMARTCROP_MOVEMENT_DEAD_ZONE,
    SMARTCROP_SAMPLE_INTERVAL,
    SMARTCROP_STACK_REACTION_EVENT_THRESHOLD,
    SMARTCROP_STACK_REACTION_WEBCAM_RATIO,
    SMARTCROP_STACK_STRONG_REACTION_EVENT_THRESHOLD,
    SMARTCROP_STACK_STRONG_REACTION_WEBCAM_RATIO,
    SMARTCROP_STACK_WEBCAM_RATIO,
    SMARTCROP_STACK_WEBCAM_RATIO_MAX,
    SMARTCROP_STACK_WEBCAM_RATIO_MIN,
    SMARTCROP_V2_ENABLED,
    SMARTCROP_WEBCAM_AVOIDANCE_MARGIN,
    SMARTCROP_WEBCAM_PERSISTENCE,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from src.gameplay_tracker import build_stable_gameplay_track
from src.logger import info, warning
from src.smart_crop import CropPlan, analyze_smart_crop
from src.webcam_detector import detect_webcam


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass
class FocusPoint:
    time: float
    center_x: float
    center_y: float
    source: str = "motion"


@dataclass
class SmartCropV2Plan:
    mode: str
    input_width: int
    input_height: int
    duration: float
    legacy_plan: CropPlan
    confidence: float = 0.0
    webcam_region: Optional[Rect] = None
    gameplay_region: Optional[Rect] = None
    gameplay_crop_width: int = 0
    gameplay_crop_height: int = 0
    gameplay_output_height: int = 0
    webcam_output_height: int = 0
    focus_points: list[FocusPoint] = field(default_factory=list)
    reaction_events: list[dict] = field(default_factory=list)
    tracking_debug: list[dict] = field(default_factory=list)
    tracking_summary: dict = field(default_factory=dict)
    webcam_detection_debug: dict = field(default_factory=dict)
    webcam_candidates: list[dict] = field(default_factory=list)
    layout_metadata: dict = field(default_factory=dict)
    decision_reason: str = ""


def _even(value: float) -> int:
    result = max(2, int(round(value)))
    return result if result % 2 == 0 else result - 1


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def validate_crop_bounds(rect: Rect, width: int, height: int) -> Rect:
    x = max(0, min(int(rect.x), max(0, width - 2)))
    y = max(0, min(int(rect.y), max(0, height - 2)))
    w = max(2, min(int(rect.w), width - x))
    h = max(2, min(int(rect.h), height - y))
    return Rect(x=x, y=y, w=w, h=h)


def _load_face_detector():
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(path))
    return None if detector.empty() else detector


_FACE_DETECTOR = _load_face_detector()


def _detect_faces(frame) -> list[tuple[int, int, int, int]]:
    if _FACE_DETECTOR is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, 640.0 / max(1.0, float(width)))
    small = (
        cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    faces = _FACE_DETECTOR.detectMultiScale(
        small,
        scaleFactor=1.10,
        minNeighbors=5,
        minSize=(32, 32),
    )
    if scale == 1.0:
        return [tuple(map(int, face)) for face in faces]
    return [
        (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        for x, y, w, h in faces
    ]


def infer_persistent_webcam(
    detections: list[list[tuple[int, int, int, int]]],
    width: int,
    height: int,
) -> tuple[Rect | None, float, str | None]:
    """Backward-compatible public helper, now powered by the scored detector.

    Existing callers historically interpret confidence as face persistence, so
    this wrapper keeps that meaning while detect_content_layout uses final_score.
    """
    samples = [(float(index), None, faces) for index, faces in enumerate(detections or [])]
    result = detect_webcam(samples, width, height)
    candidate = result.selected
    if candidate is None:
        top = result.candidates[0] if result.candidates else None
        confidence = top.face_persistence_score if top else 0.0
        corner = top.corner if top else None
        return None, confidence, corner
    roi = candidate.roi
    rect = validate_crop_bounds(Rect(roi.x, roi.y, roi.w, roi.h), width, height)
    if candidate.face_persistence_score < SMARTCROP_WEBCAM_PERSISTENCE:
        return None, candidate.face_persistence_score, candidate.corner
    return rect, candidate.face_persistence_score, candidate.corner


def _calculate_crop_size_for_aspect(width: int, height: int, target_ratio: float) -> tuple[int, int]:
    source_ratio = width / max(1.0, float(height))
    if source_ratio >= target_ratio:
        crop_h = height
        crop_w = _even(height * target_ratio)
    else:
        crop_w = width
        crop_h = _even(width / target_ratio)
    return min(width, crop_w), min(height, crop_h)


def smooth_focus_points(
    points: list[FocusPoint],
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> list[FocusPoint]:
    """Legacy-compatible smoother kept for tests/fallback callers."""
    if not points:
        return []
    result = [points[0]]
    dead_x = crop_width * SMARTCROP_MOVEMENT_DEAD_ZONE
    dead_y = crop_height * SMARTCROP_MOVEMENT_DEAD_ZONE
    max_x = crop_width * SMARTCROP_MAX_CROP_VELOCITY
    max_y = crop_height * SMARTCROP_MAX_CROP_VELOCITY

    for point in points[1:]:
        previous = result[-1]
        dx = point.center_x - previous.center_x
        dy = point.center_y - previous.center_y
        if abs(dx) <= dead_x:
            dx = 0.0
        if abs(dy) <= dead_y:
            dy = 0.0
        dx = _clamp(dx, -max_x, max_x)
        dy = _clamp(dy, -max_y, max_y)
        cx = previous.center_x + dx * 0.38
        cy = previous.center_y + dy * 0.38
        cx = _clamp(cx, crop_width / 2, frame_width - crop_width / 2)
        cy = _clamp(cy, crop_height / 2, frame_height - crop_height / 2)
        result.append(FocusPoint(point.time, cx, cy, point.source))
    return result


def _sample_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Nu pot deschide video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 1.0

    samples = []
    position = 0.0
    try:
        while position <= duration + 0.001:
            capture.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0)
            ok, frame = capture.read()
            if not ok:
                break
            samples.append((position, frame, _detect_faces(frame)))
            position += SMARTCROP_SAMPLE_INTERVAL
    finally:
        capture.release()
    return width, height, duration, samples


def _detect_reaction_events(samples, webcam: Rect) -> list[dict]:
    previous_center = None
    previous_area = None
    events: list[dict] = []
    for time_position, _frame, faces in samples:
        inside = []
        for face in faces or []:
            x, y, w, h = face
            cx = x + w / 2.0
            cy = y + h / 2.0
            if webcam.x <= cx <= webcam.x2 and webcam.y <= cy <= webcam.y2:
                inside.append(face)
        if not inside:
            continue
        face = max(inside, key=lambda box: box[2] * box[3])
        center = (face[0] + face[2] / 2.0, face[1] + face[3] / 2.0)
        area = float(face[2] * face[3])
        if previous_center is not None:
            movement = ((center[0] - previous_center[0]) ** 2 + (center[1] - previous_center[1]) ** 2) ** 0.5
            if movement > max(face[2], face[3]) * 0.42:
                events.append({"time": round(time_position, 3), "type": "face_motion_spike"})
        if previous_area is not None and previous_area > 0:
            area_change = abs(area - previous_area) / previous_area
            if area_change >= 0.32:
                events.append({"time": round(time_position, 3), "type": "face_scale_spike"})
        previous_center = center
        previous_area = area
    return events


def choose_stack_webcam_ratio(reaction_events: list[dict]) -> float:
    count = len(reaction_events or [])
    if count >= SMARTCROP_STACK_STRONG_REACTION_EVENT_THRESHOLD:
        ratio = SMARTCROP_STACK_STRONG_REACTION_WEBCAM_RATIO
    elif count >= SMARTCROP_STACK_REACTION_EVENT_THRESHOLD:
        ratio = SMARTCROP_STACK_REACTION_WEBCAM_RATIO
    else:
        ratio = SMARTCROP_STACK_WEBCAM_RATIO
    return _clamp(ratio, SMARTCROP_STACK_WEBCAM_RATIO_MIN, SMARTCROP_STACK_WEBCAM_RATIO_MAX)


def _build_layout_metadata(webcam_ratio: float) -> dict:
    webcam_height = _even(VIDEO_HEIGHT * webcam_ratio)
    webcam_height = max(2, min(VIDEO_HEIGHT - 2, webcam_height))
    gameplay_height = VIDEO_HEIGHT - webcam_height
    return {
        "type": "GAMEPLAY_WEBCAM_STACK",
        "webcam_position": "top",
        "gameplay_position": "bottom",
        "webcam_ratio": round(webcam_height / VIDEO_HEIGHT, 4),
        "gameplay_ratio": round(gameplay_height / VIDEO_HEIGHT, 4),
        "split_y": webcam_height,
        "webcam_height": webcam_height,
        "gameplay_height": gameplay_height,
        "blurred_fill": bool(SMARTCROP_BLURRED_FILL_ENABLED),
        "caption_safe_area": {
            "x1": int(CAPTION_MARGIN_L),
            "x2": int(VIDEO_WIDTH - CAPTION_MARGIN_R),
            "y1": int(min(VIDEO_HEIGHT - 120, webcam_height + 70)),
            "y2": int(min(VIDEO_HEIGHT - 120, webcam_height + max(260, gameplay_height * 0.42))),
        },
    }


def _log_webcam_detection(result) -> None:
    if not result.candidates:
        info(f"[WebcamDetector] selected=false reason={result.reason}")
        return
    top = result.candidates[0]
    info(
        "[WebcamDetector] "
        f"face_presence={top.face_presence_score:.2f} | "
        f"face_persistence={top.face_persistence_score:.2f} | "
        f"roi_stability={top.roi_stability_score:.2f} | "
        f"geometric_consistency={top.geometric_consistency_score:.2f} | "
        f"overlay_likelihood={top.overlay_likelihood_score:.2f} | "
        f"visual_separation={top.visual_separation_score:.2f} | "
        f"temporal_separation={top.temporal_separation_score:.2f} | "
        f"false_positive_penalty={top.false_positive_penalty:.2f} | "
        f"final_score={top.final_score:.2f}"
    )
    info(
        f"[WebcamDetector] candidates={len(result.candidates)} | "
        f"ambiguous={result.ambiguous} | selected={result.selected is not None} | reason={result.reason}"
    )


def detect_content_layout(video_path: Path) -> SmartCropV2Plan:
    legacy = analyze_smart_crop(video_path)
    if not SMARTCROP_V2_ENABLED:
        return SmartCropV2Plan("GENERAL", legacy.input_width, legacy.input_height, legacy.duration, legacy)

    try:
        width, height, duration, samples = _sample_video(video_path)
        if not samples:
            raise RuntimeError("Nu există frame-uri pentru analiza SmartCrop 2.0")

        detection = detect_webcam(samples, width, height)
        _log_webcam_detection(detection)
        candidates_debug = [candidate.to_debug_dict() for candidate in detection.candidates]

        if (
            detection.selected is None
            or not SMARTCROP_DETECT_GAMEPLAY_WEBCAM
            or width <= height
        ):
            confidence = detection.candidates[0].final_score if detection.candidates else 0.0
            return SmartCropV2Plan(
                mode="GENERAL",
                input_width=width,
                input_height=height,
                duration=duration,
                legacy_plan=legacy,
                confidence=confidence,
                webcam_detection_debug={
                    "selected": False,
                    "reason": detection.reason,
                    "ambiguous": detection.ambiguous,
                },
                webcam_candidates=candidates_debug,
                decision_reason="robust webcam detector did not confirm an overlay",
            )

        selected = detection.selected
        webcam = validate_crop_bounds(
            Rect(selected.roi.x, selected.roi.y, selected.roi.w, selected.roi.h),
            width,
            height,
        )
        reaction_events = _detect_reaction_events(samples, webcam)
        webcam_ratio = choose_stack_webcam_ratio(reaction_events)
        layout = _build_layout_metadata(webcam_ratio)
        webcam_output_height = int(layout["webcam_height"])
        gameplay_output_height = int(layout["gameplay_height"])

        gameplay_target_ratio = VIDEO_WIDTH / float(gameplay_output_height)
        crop_w, crop_h = _calculate_crop_size_for_aspect(width, height, gameplay_target_ratio)
        track, tracking_summary = build_stable_gameplay_track(
            samples,
            crop_width=crop_w,
            crop_height=crop_h,
            frame_width=width,
            frame_height=height,
            ignored_region=webcam,
        )
        focus_points = [
            FocusPoint(item.time, item.center_x, item.center_y, item.source)
            for item in track
        ]

        info(
            "[LayoutPolicy] type=GAMEPLAY_WEBCAM_STACK | webcam_position=top | "
            f"gameplay_position=bottom | webcam_ratio={layout['webcam_ratio']:.2f} | "
            f"gameplay_ratio={layout['gameplay_ratio']:.2f} | split_y={layout['split_y']}"
        )
        info(
            f"[SMARTCROP] Webcam bbox: x={webcam.x} y={webcam.y} w={webcam.w} h={webcam.h} | "
            f"source_position={selected.corner}"
        )
        info(
            "[SMARTCROP] Stable gameplay tracker: "
            f"samples={len(focus_points)} switches={tracking_summary.get('target_switches', 0)} "
            f"direction_changes={tracking_summary.get('direction_changes', 0)}"
        )

        return SmartCropV2Plan(
            mode="GAMEPLAY_WEBCAM_STACK",
            input_width=width,
            input_height=height,
            duration=duration,
            legacy_plan=legacy,
            confidence=selected.final_score,
            webcam_region=webcam,
            gameplay_region=Rect(0, 0, width, height),
            gameplay_crop_width=crop_w,
            gameplay_crop_height=crop_h,
            gameplay_output_height=gameplay_output_height,
            webcam_output_height=webcam_output_height,
            focus_points=focus_points,
            reaction_events=reaction_events,
            tracking_debug=[item.debug for item in track],
            tracking_summary=tracking_summary,
            webcam_detection_debug={
                **selected.to_debug_dict(),
                "selected": True,
                "ambiguous": detection.ambiguous,
                "reason": detection.reason,
            },
            webcam_candidates=candidates_debug,
            layout_metadata=layout,
            decision_reason="robust scored webcam detector + top webcam stack + stable gameplay tracking",
        )
    except Exception as exc:
        warning(f"[SMARTCROP] V2 analysis failed: {exc}; fallback la SmartCrop existent.")
        return SmartCropV2Plan(
            mode="GENERAL",
            input_width=legacy.input_width,
            input_height=legacy.input_height,
            duration=legacy.duration,
            legacy_plan=legacy,
            decision_reason=f"SmartCrop V2 analysis failed: {exc}",
        )


def analyze_smart_crop_v2(video_path: Path) -> SmartCropV2Plan:
    return detect_content_layout(Path(video_path))


def _focus_xy(point: FocusPoint, plan: SmartCropV2Plan) -> tuple[float, float]:
    max_x = max(0, plan.input_width - plan.gameplay_crop_width)
    max_y = max(0, plan.input_height - plan.gameplay_crop_height)
    x = _clamp(point.center_x - plan.gameplay_crop_width / 2, 0, max_x)
    y = _clamp(point.center_y - plan.gameplay_crop_height / 2, 0, max_y)

    # Avoid duplicating the original webcam overlay inside the gameplay panel
    # when a small horizontal shift can remove it without breaking bounds.
    webcam = plan.webcam_region
    if webcam is not None and plan.mode in {"GAMEPLAY_WEBCAM", "GAMEPLAY_WEBCAM_STACK"}:
        margin = max(0, int(SMARTCROP_WEBCAM_AVOIDANCE_MARGIN))
        webcam_center_x = webcam.x + webcam.w / 2.0
        if webcam_center_x < plan.input_width / 2.0:
            safe_x = webcam.x2 + margin
            if safe_x <= max_x:
                x = max(x, safe_x)
        else:
            safe_x = webcam.x - plan.gameplay_crop_width - margin
            if safe_x >= 0:
                x = min(x, safe_x)
    return x, y


def initial_gameplay_xy(plan: SmartCropV2Plan) -> tuple[float, float]:
    if not plan.focus_points:
        return (
            max(0.0, (plan.input_width - plan.gameplay_crop_width) / 2),
            max(0.0, (plan.input_height - plan.gameplay_crop_height) / 2),
        )
    return _focus_xy(plan.focus_points[0], plan)


def write_gameplay_sendcmd(
    plan: SmartCropV2Plan,
    output_file: Path,
    target_name: str = "gameplay",
) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    points = plan.focus_points
    lines: list[str] = []

    if len(points) <= 1:
        x, y = initial_gameplay_xy(plan)
        lines.append(f"0.000 crop@{target_name} x {x:.3f};")
        lines.append(f"0.000 crop@{target_name} y {y:.3f};")
    else:
        for current, nxt in zip(points, points[1:]):
            start = current.time
            end = max(nxt.time, start + 0.001)
            x1, y1 = _focus_xy(current, plan)
            x2, y2 = _focus_xy(nxt, plan)
            lines.append(
                f"{start:.3f}-{end:.3f} [expr] crop@{target_name} x 'lerp({x1:.3f},{x2:.3f},TI)';"
            )
            lines.append(
                f"{start:.3f}-{end:.3f} [expr] crop@{target_name} y 'lerp({y1:.3f},{y2:.3f},TI)';"
            )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_file


def validate_layout_plan(plan: SmartCropV2Plan) -> bool:
    if plan.mode not in {"GAMEPLAY_WEBCAM", "GAMEPLAY_WEBCAM_STACK"}:
        return True
    if plan.webcam_region is None:
        return False
    if plan.gameplay_crop_width <= 0 or plan.gameplay_crop_height <= 0:
        return False
    if plan.gameplay_output_height + plan.webcam_output_height != VIDEO_HEIGHT:
        return False
    if plan.mode == "GAMEPLAY_WEBCAM_STACK":
        layout = plan.layout_metadata or {}
        if layout.get("webcam_position") != "top" or layout.get("gameplay_position") != "bottom":
            return False
        if int(layout.get("split_y", -1)) != plan.webcam_output_height:
            return False
    return True


def save_smartcrop_debug(video_name: str, plan: SmartCropV2Plan) -> None:
    if not SMARTCROP_DEBUG:
        return
    output_dir = HIGHLIGHTS_DIR / "debug" / video_name / "smartcrop"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": plan.mode,
        "confidence": plan.confidence,
        "decision_reason": plan.decision_reason,
        "input": [plan.input_width, plan.input_height],
        "webcam": (
            [plan.webcam_region.x, plan.webcam_region.y, plan.webcam_region.w, plan.webcam_region.h]
            if plan.webcam_region else None
        ),
        "webcam_detector": plan.webcam_detection_debug,
        "candidate_webcam_regions": plan.webcam_candidates,
        "gameplay_crop": [plan.gameplay_crop_width, plan.gameplay_crop_height],
        "layout": plan.layout_metadata or {
            "gameplay_height": plan.gameplay_output_height,
            "webcam_height": plan.webcam_output_height,
        },
        "reaction_events": plan.reaction_events,
        "tracking_summary": plan.tracking_summary,
        "tracking_samples": plan.tracking_debug,
        "focus_points": [
            {
                "time": round(point.time, 3),
                "center_x": round(point.center_x, 2),
                "center_y": round(point.center_y, 2),
                "source": point.source,
            }
            for point in plan.focus_points
        ],
    }
    (output_dir / "plan.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
