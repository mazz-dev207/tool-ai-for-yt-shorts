from __future__ import annotations

from dataclasses import dataclass
import math
import os

import cv2
import numpy as np


READING_FOCUS_ENABLED = os.getenv("READING_FOCUS_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
READING_FOCUS_MIN_SCORE = float(os.getenv("READING_FOCUS_MIN_SCORE", "0.58"))
READING_FOCUS_MIN_STREAK = max(2, int(os.getenv("READING_FOCUS_MIN_STREAK", "2")))
READING_FOCUS_BLEND = float(os.getenv("READING_FOCUS_BLEND", "0.34"))
READING_FOCUS_RELEASE_BLEND = float(os.getenv("READING_FOCUS_RELEASE_BLEND", "0.30"))
READING_FOCUS_MAX_SHIFT_X = float(os.getenv("READING_FOCUS_MAX_SHIFT_X", "0.20"))
READING_FOCUS_MAX_SHIFT_Y = float(os.getenv("READING_FOCUS_MAX_SHIFT_Y", "0.16"))


@dataclass(frozen=True)
class ReadingCandidate:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2.0


@dataclass(frozen=True)
class ReadingTarget:
    time: float
    center_x: float
    center_y: float
    confidence: float
    bbox: tuple[int, int, int, int]
    streak: int
    low_motion_score: float


@dataclass(frozen=True)
class AdjustedFocusPoint:
    time: float
    center_x: float
    center_y: float
    source: str
    debug: dict


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _rect_overlap_ratio(candidate: ReadingCandidate, ignored_region) -> float:
    if ignored_region is None:
        return 0.0
    ix1 = max(candidate.x, int(getattr(ignored_region, "x", 0)))
    iy1 = max(candidate.y, int(getattr(ignored_region, "y", 0)))
    ix2 = min(candidate.x2, int(getattr(ignored_region, "x2", getattr(ignored_region, "x", 0) + getattr(ignored_region, "w", 0))))
    iy2 = min(candidate.y2, int(getattr(ignored_region, "y2", getattr(ignored_region, "y", 0) + getattr(ignored_region, "h", 0))))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = float((ix2 - ix1) * (iy2 - iy1))
    return intersection / max(1.0, float(candidate.w * candidate.h))


def _prepare_gray(frame, max_width: int = 960):
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


def _candidate_score(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    frame_h, frame_w = gray.shape[:2]
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0

    edges = cv2.Canny(roi, 55, 145)
    edge_density = float(np.count_nonzero(edges)) / max(1.0, float(edges.size))
    edge_score = _clamp(edge_density / 0.18)

    width_ratio = w / max(1.0, float(frame_w))
    height_ratio = h / max(1.0, float(frame_h))
    area_ratio = width_ratio * height_ratio
    aspect = w / max(1.0, float(h))

    geometry_score = _clamp(
        0.45 * min(1.0, aspect / 3.0)
        + 0.30 * min(1.0, width_ratio / 0.34)
        + 0.25 * min(1.0, height_ratio / 0.10)
    )

    area_score = 1.0 - min(1.0, abs(area_ratio - 0.055) / 0.10)
    cx = x + w / 2.0
    cy = y + h / 2.0
    nx = abs(cx - frame_w / 2.0) / max(1.0, frame_w / 2.0)
    ny = abs(cy - frame_h / 2.0) / max(1.0, frame_h / 2.0)
    centrality = _clamp(1.0 - math.hypot(nx * 0.75, ny * 0.55) / 1.25)

    corner_penalty = 0.0
    near_corner = nx > 0.72 and ny > 0.58
    if near_corner and area_ratio < 0.025:
        corner_penalty = 0.28
    if width_ratio < 0.12 or height_ratio < 0.025:
        corner_penalty += 0.18

    return _clamp(
        0.38 * edge_score
        + 0.27 * geometry_score
        + 0.18 * area_score
        + 0.17 * centrality
        - corner_penalty
    )


def detect_text_candidates(frame, ignored_region=None) -> list[ReadingCandidate]:
    """Detect large text/UI-like regions without requiring OCR.

    The detector intentionally prefers menu/dialog/chat-size blocks over tiny HUD
    labels. It does not claim to understand the words; it finds likely readable
    screen regions that can become a temporary virtual-camera target.
    """
    if frame is None or not hasattr(frame, "shape"):
        return []

    gray, scale = _prepare_gray(frame)
    height, width = gray.shape[:2]

    gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient = cv2.convertScaleAbs(gradient)
    gradient = cv2.GaussianBlur(gradient, (3, 3), 0)
    _threshold, binary = cv2.threshold(
        gradient,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    kernel_w = max(9, int(round(width * 0.026)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    merged = cv2.dilate(
        merged,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    )

    contours, _hierarchy = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    inverse = 1.0 / max(scale, 1e-6)
    frame_h, frame_w = frame.shape[:2]
    candidates: list[ReadingCandidate] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        width_ratio = w / max(1.0, float(width))
        height_ratio = h / max(1.0, float(height))
        area_ratio = width_ratio * height_ratio
        aspect = w / max(1.0, float(h))

        if not (0.10 <= width_ratio <= 0.90):
            continue
        if not (0.022 <= height_ratio <= 0.34):
            continue
        if not (0.0018 <= area_ratio <= 0.22):
            continue
        if not (1.25 <= aspect <= 24.0):
            continue

        confidence = _candidate_score(gray, x, y, w, h)
        if confidence < 0.34:
            continue

        candidate = ReadingCandidate(
            x=max(0, int(round(x * inverse))),
            y=max(0, int(round(y * inverse))),
            w=max(2, int(round(w * inverse))),
            h=max(2, int(round(h * inverse))),
            confidence=confidence,
        )
        if candidate.x2 > frame_w or candidate.y2 > frame_h:
            candidate = ReadingCandidate(
                x=min(frame_w - 2, candidate.x),
                y=min(frame_h - 2, candidate.y),
                w=max(2, min(candidate.w, frame_w - candidate.x)),
                h=max(2, min(candidate.h, frame_h - candidate.y)),
                confidence=candidate.confidence,
            )
        if _rect_overlap_ratio(candidate, ignored_region) > 0.20:
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates[:8]


def _same_region(a: ReadingCandidate | None, b: ReadingCandidate | None, frame_width: int, frame_height: int) -> bool:
    if a is None or b is None:
        return False
    diagonal = max(1.0, math.hypot(frame_width, frame_height))
    distance = math.hypot(a.center_x - b.center_x, a.center_y - b.center_y) / diagonal
    size_ratio = max(
        a.w / max(1.0, b.w),
        b.w / max(1.0, a.w),
        a.h / max(1.0, b.h),
        b.h / max(1.0, a.h),
    )
    return distance <= 0.11 and size_ratio <= 1.85


def _motion_score(previous_frame, current_frame, candidate: ReadingCandidate) -> float:
    if previous_frame is None or current_frame is None:
        return 0.55
    fh, fw = current_frame.shape[:2]
    x1 = max(0, min(fw - 1, candidate.x))
    y1 = max(0, min(fh - 1, candidate.y))
    x2 = max(x1 + 1, min(fw, candidate.x2))
    y2 = max(y1 + 1, min(fh, candidate.y2))
    previous = previous_frame[y1:y2, x1:x2]
    current = current_frame[y1:y2, x1:x2]
    if previous.size == 0 or current.size == 0 or previous.shape != current.shape:
        return 0.50
    previous_gray = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (120, 72))
    current_gray = cv2.resize(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (120, 72))
    motion = float(cv2.absdiff(previous_gray, current_gray).mean()) / 255.0
    return _clamp(1.0 - motion / 0.18)


def detect_reading_targets(samples, ignored_region=None) -> list[ReadingTarget | None]:
    if not READING_FOCUS_ENABLED:
        return [None for _sample in samples or []]

    targets: list[ReadingTarget | None] = []
    previous_candidate: ReadingCandidate | None = None
    previous_frame = None
    streak = 0

    for sample in samples or []:
        time_position = float(sample[0])
        frame = sample[1]
        frame_h, frame_w = frame.shape[:2]
        candidates = detect_text_candidates(frame, ignored_region=ignored_region)

        selected = None
        if candidates:
            if previous_candidate is not None:
                ranked = sorted(
                    candidates,
                    key=lambda item: (
                        item.confidence + (0.20 if _same_region(item, previous_candidate, frame_w, frame_h) else 0.0)
                    ),
                    reverse=True,
                )
                selected = ranked[0]
            else:
                selected = candidates[0]

        if selected is not None and _same_region(selected, previous_candidate, frame_w, frame_h):
            streak += 1
        elif selected is not None:
            streak = 1
        else:
            streak = 0

        if selected is None:
            targets.append(None)
            previous_candidate = None
            previous_frame = frame
            continue

        low_motion = _motion_score(previous_frame, frame, selected)
        persistence = _clamp(streak / max(1.0, float(READING_FOCUS_MIN_STREAK + 1)))
        reading_score = _clamp(
            0.56 * selected.confidence
            + 0.26 * persistence
            + 0.18 * low_motion
        )

        active = streak >= READING_FOCUS_MIN_STREAK and reading_score >= READING_FOCUS_MIN_SCORE
        if active:
            targets.append(
                ReadingTarget(
                    time=time_position,
                    center_x=selected.center_x,
                    center_y=selected.center_y,
                    confidence=reading_score,
                    bbox=(selected.x, selected.y, selected.w, selected.h),
                    streak=streak,
                    low_motion_score=low_motion,
                )
            )
        else:
            targets.append(None)

        previous_candidate = selected
        previous_frame = frame

    return targets


def _camera_center(x: float, y: float, crop_width: int, crop_height: int, frame_width: int, frame_height: int) -> tuple[float, float]:
    return (
        _clamp(x, crop_width / 2.0, frame_width - crop_width / 2.0),
        _clamp(y, crop_height / 2.0, frame_height - crop_height / 2.0),
    )


def apply_reading_focus(
    focus_points,
    samples,
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
    ignored_region=None,
) -> tuple[list[AdjustedFocusPoint], dict]:
    """Blend stable gameplay tracking with persistent on-screen reading targets.

    Reading focus is deliberately a temporary semantic override. It never snaps:
    movement is rate-limited and releases gradually back to the gameplay track.
    """
    if not focus_points or not samples or not READING_FOCUS_ENABLED:
        adjusted = [
            AdjustedFocusPoint(
                time=float(point.time),
                center_x=float(point.center_x),
                center_y=float(point.center_y),
                source=str(getattr(point, "source", "gameplay")),
                debug={"READING_FOCUS": False},
            )
            for point in focus_points or []
        ]
        return adjusted, {
            "reading_focus_events": 0,
            "reading_focus_samples": 0,
            "reading_focus_max_confidence": 0.0,
        }

    targets = detect_reading_targets(samples, ignored_region=ignored_region)
    sample_by_time = {
        round(float(sample[0]), 3): index for index, sample in enumerate(samples)
    }

    current_x = float(focus_points[0].center_x)
    current_y = float(focus_points[0].center_y)
    adjusted: list[AdjustedFocusPoint] = []
    active_samples = 0
    events = 0
    max_confidence = 0.0
    was_reading = False

    max_shift_x = crop_width * READING_FOCUS_MAX_SHIFT_X
    max_shift_y = crop_height * READING_FOCUS_MAX_SHIFT_Y

    for point in focus_points:
        key = round(float(point.time), 3)
        index = sample_by_time.get(key)
        target = targets[index] if index is not None and index < len(targets) else None

        base_x = float(point.center_x)
        base_y = float(point.center_y)
        reading = target is not None

        if reading:
            if not was_reading:
                events += 1
            active_samples += 1
            max_confidence = max(max_confidence, target.confidence)
            desired_x, desired_y = _camera_center(
                target.center_x,
                target.center_y,
                crop_width,
                crop_height,
                frame_width,
                frame_height,
            )
            blend = _clamp(READING_FOCUS_BLEND, 0.05, 0.80)
            dx = _clamp((desired_x - current_x) * blend, -max_shift_x, max_shift_x)
            dy = _clamp((desired_y - current_y) * blend, -max_shift_y, max_shift_y)
            current_x += dx
            current_y += dy
            source = "reading_focus"
            debug = {
                "READING_FOCUS": True,
                "READING_CONFIDENCE": round(target.confidence, 4),
                "READING_BBOX": list(target.bbox),
                "READING_STREAK": target.streak,
                "READING_LOW_MOTION": round(target.low_motion_score, 4),
                "BASE_GAMEPLAY_CENTER": [round(base_x, 2), round(base_y, 2)],
            }
        else:
            release = _clamp(READING_FOCUS_RELEASE_BLEND, 0.05, 0.80)
            dx = _clamp((base_x - current_x) * release, -max_shift_x, max_shift_x)
            dy = _clamp((base_y - current_y) * release, -max_shift_y, max_shift_y)
            current_x += dx
            current_y += dy
            source = str(getattr(point, "source", "gameplay"))
            debug = {
                "READING_FOCUS": False,
                "BASE_GAMEPLAY_CENTER": [round(base_x, 2), round(base_y, 2)],
            }

        current_x, current_y = _camera_center(
            current_x,
            current_y,
            crop_width,
            crop_height,
            frame_width,
            frame_height,
        )
        adjusted.append(
            AdjustedFocusPoint(
                time=float(point.time),
                center_x=current_x,
                center_y=current_y,
                source=source,
                debug=debug,
            )
        )
        was_reading = reading

    return adjusted, {
        "reading_focus_events": events,
        "reading_focus_samples": active_samples,
        "reading_focus_max_confidence": round(max_confidence, 4),
    }
