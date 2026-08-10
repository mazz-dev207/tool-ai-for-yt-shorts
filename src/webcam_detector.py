from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import (
    SMARTCROP_WEBCAM_AMBIGUITY_MARGIN,
    SMARTCROP_WEBCAM_AMBIGUOUS_MIN_PERSISTENCE,
    SMARTCROP_WEBCAM_AMBIGUOUS_MIN_STABILITY,
    SMARTCROP_WEBCAM_MAX_AREA_RATIO,
    SMARTCROP_WEBCAM_MIN_AREA_RATIO,
    SMARTCROP_WEBCAM_SCORE_THRESHOLD,
)


@dataclass(frozen=True)
class WebcamROI:
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
class WebcamCandidate:
    roi: WebcamROI
    corner: str
    face_presence_score: float
    face_persistence_score: float
    roi_stability_score: float
    geometric_consistency_score: float
    overlay_likelihood_score: float
    visual_separation_score: float
    temporal_separation_score: float
    false_positive_penalty: float
    final_score: float
    hits: int
    total_samples: int
    sample_indices: list[int] = field(default_factory=list)

    def to_debug_dict(self) -> dict:
        return {
            "roi": [self.roi.x, self.roi.y, self.roi.w, self.roi.h],
            "corner": self.corner,
            "face_presence": round(self.face_presence_score, 4),
            "face_persistence": round(self.face_persistence_score, 4),
            "roi_stability": round(self.roi_stability_score, 4),
            "geometric_consistency": round(self.geometric_consistency_score, 4),
            "overlay_likelihood": round(self.overlay_likelihood_score, 4),
            "visual_separation": round(self.visual_separation_score, 4),
            "temporal_separation": round(self.temporal_separation_score, 4),
            "false_positive_penalty": round(self.false_positive_penalty, 4),
            "final_score": round(self.final_score, 4),
            "hits": self.hits,
            "total_samples": self.total_samples,
        }


@dataclass
class WebcamDetectionResult:
    selected: WebcamCandidate | None
    candidates: list[WebcamCandidate]
    ambiguous: bool = False
    reason: str = ""


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_median(values, default=0.0) -> float:
    return float(statistics.median(values)) if values else float(default)


def _corner_for_center(cx: float, cy: float, width: int, height: int) -> str:
    horizontal = "left" if cx < width / 2 else "right"
    vertical = "top" if cy < height / 2 else "bottom"
    return f"{vertical}-{horizontal}"


def _near_edge_score(cx: float, cy: float, width: int, height: int) -> float:
    nx = min(cx / max(1.0, width), (width - cx) / max(1.0, width))
    ny = min(cy / max(1.0, height), (height - cy) / max(1.0, height))
    nearest = min(nx, ny)
    return _clamp(1.0 - nearest / 0.34)


def _cluster_face_detections(samples, width: int, height: int) -> list[dict]:
    diagonal = max(1.0, math.hypot(width, height))
    frame_area = max(1.0, float(width * height))
    clusters: list[dict] = []

    for sample_index, sample in enumerate(samples or []):
        faces = sample[2] if len(sample) >= 3 else []
        for raw in faces or []:
            try:
                x, y, w, h = [float(value) for value in raw]
            except (TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue

            face_area_ratio = (w * h) / frame_area
            min_face_area = max(0.00035, SMARTCROP_WEBCAM_MIN_AREA_RATIO * 0.20)
            max_face_area = min(0.16, max(0.06, SMARTCROP_WEBCAM_MAX_AREA_RATIO))
            if not (min_face_area <= face_area_ratio <= max_face_area):
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0
            best_cluster = None
            best_distance = float("inf")
            for cluster in clusters:
                mean_cx = statistics.fmean(cluster["centers_x"])
                mean_cy = statistics.fmean(cluster["centers_y"])
                mean_w = statistics.fmean(cluster["widths"])
                mean_h = statistics.fmean(cluster["heights"])
                spatial = math.hypot(cx - mean_cx, cy - mean_cy) / diagonal
                size_ratio = max(
                    w / max(1.0, mean_w),
                    mean_w / max(1.0, w),
                    h / max(1.0, mean_h),
                    mean_h / max(1.0, h),
                )
                if spatial <= 0.105 and size_ratio <= 1.90 and spatial < best_distance:
                    best_cluster = cluster
                    best_distance = spatial

            if best_cluster is None:
                best_cluster = {
                    "boxes": [],
                    "centers_x": [],
                    "centers_y": [],
                    "widths": [],
                    "heights": [],
                    "sample_indices": set(),
                }
                clusters.append(best_cluster)

            best_cluster["boxes"].append((x, y, w, h))
            best_cluster["centers_x"].append(cx)
            best_cluster["centers_y"].append(cy)
            best_cluster["widths"].append(w)
            best_cluster["heights"].append(h)
            best_cluster["sample_indices"].add(sample_index)

    return clusters


def _roi_from_cluster(cluster: dict, width: int, height: int) -> WebcamROI:
    median_x = _safe_median([box[0] for box in cluster["boxes"]])
    median_y = _safe_median([box[1] for box in cluster["boxes"]])
    median_w = _safe_median(cluster["widths"], 80.0)
    median_h = _safe_median(cluster["heights"], 80.0)

    # Asymmetric padding keeps headroom while including shoulders/hands.
    pad_left = median_w * 1.05
    pad_right = median_w * 1.05
    pad_top = median_h * 0.70
    pad_bottom = median_h * 1.55

    x = max(0, int(round(median_x - pad_left)))
    y = max(0, int(round(median_y - pad_top)))
    x2 = min(width, int(round(median_x + median_w + pad_right)))
    y2 = min(height, int(round(median_y + median_h + pad_bottom)))
    return WebcamROI(x=x, y=y, w=max(2, x2 - x), h=max(2, y2 - y))


def _stability_scores(cluster: dict, width: int, height: int) -> tuple[float, float]:
    centers_x = cluster["centers_x"]
    centers_y = cluster["centers_y"]
    widths = cluster["widths"]
    heights = cluster["heights"]

    center_jitter = max(
        statistics.pstdev(centers_x) / max(1.0, width) if len(centers_x) >= 2 else 0.0,
        statistics.pstdev(centers_y) / max(1.0, height) if len(centers_y) >= 2 else 0.0,
    )
    mean_w = max(1.0, statistics.fmean(widths))
    mean_h = max(1.0, statistics.fmean(heights))
    size_jitter = max(
        statistics.pstdev(widths) / mean_w if len(widths) >= 2 else 0.0,
        statistics.pstdev(heights) / mean_h if len(heights) >= 2 else 0.0,
    )
    roi_stability = _clamp(1.0 - center_jitter / 0.085 * 0.62 - size_jitter / 0.40 * 0.38)

    aspect_ratios = [w / max(1.0, h) for w, h in zip(widths, heights)]
    area_values = [w * h for w, h in zip(widths, heights)]
    aspect_jitter = (
        statistics.pstdev(aspect_ratios) / max(0.01, statistics.fmean(aspect_ratios))
        if len(aspect_ratios) >= 2 else 0.0
    )
    area_jitter = (
        statistics.pstdev(area_values) / max(1.0, statistics.fmean(area_values))
        if len(area_values) >= 2 else 0.0
    )
    geometry = _clamp(1.0 - aspect_jitter / 0.35 * 0.55 - area_jitter / 0.60 * 0.45)
    return roi_stability, geometry


def _visual_separation(samples, roi: WebcamROI) -> tuple[float, float]:
    visual_scores: list[float] = []
    roi_motion: list[float] = []
    global_motion: list[float] = []
    previous_roi_gray = None
    previous_global_gray = None

    for sample in samples or []:
        frame = sample[1] if len(sample) >= 2 else None
        if frame is None or not hasattr(frame, "shape"):
            continue
        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 1, roi.x))
        y1 = max(0, min(fh - 1, roi.y))
        x2 = max(x1 + 1, min(fw, roi.x2))
        y2 = max(y1 + 1, min(fh, roi.y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_mean = np.array(cv2.mean(crop)[:3], dtype=np.float32)
        frame_mean = np.array(cv2.mean(frame)[:3], dtype=np.float32)
        color_distance = float(np.linalg.norm(crop_mean - frame_mean)) / 441.7

        crop_gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (96, 72))
        global_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 72))
        crop_edges = float(cv2.Canny(crop_gray, 60, 140).mean()) / 255.0
        visual_scores.append(_clamp(color_distance * 2.4 + crop_edges * 0.35))

        if previous_roi_gray is not None:
            roi_motion.append(float(cv2.absdiff(previous_roi_gray, crop_gray).mean()) / 255.0)
        if previous_global_gray is not None:
            global_motion.append(float(cv2.absdiff(previous_global_gray, global_gray).mean()) / 255.0)
        previous_roi_gray = crop_gray
        previous_global_gray = global_gray

    visual = statistics.fmean(visual_scores) if visual_scores else 0.50
    if roi_motion and global_motion:
        roi_mean = statistics.fmean(roi_motion)
        global_mean = statistics.fmean(global_motion)
        separation = abs(roi_mean - global_mean)
        temporal = _clamp(0.45 + separation * 5.0)
    else:
        temporal = 0.50
    return _clamp(visual), _clamp(temporal)


def _false_positive_penalty(
    cluster: dict,
    roi: WebcamROI,
    width: int,
    height: int,
    persistence: float,
    stability: float,
) -> float:
    frame_area = max(1.0, float(width * height))
    roi_area_ratio = (roi.w * roi.h) / frame_area
    face_area_ratio = statistics.fmean(
        [(w * h) / frame_area for w, h in zip(cluster["widths"], cluster["heights"])]
    )
    cx = roi.x + roi.w / 2.0
    cy = roi.y + roi.h / 2.0
    central_x = width * 0.28 <= cx <= width * 0.72
    central_y = height * 0.24 <= cy <= height * 0.76

    penalty = 0.0
    if persistence < 0.35:
        penalty += 0.26
    elif persistence < 0.50:
        penalty += 0.12
    if stability < 0.45:
        penalty += 0.18
    if roi_area_ratio > 0.34:
        penalty += 0.28
    elif roi_area_ratio > 0.25:
        penalty += 0.14
    if face_area_ratio > 0.10:
        penalty += 0.24  # Large cutscene/character face.
    if central_x and central_y:
        penalty += 0.16
    if _near_edge_score(cx, cy, width, height) < 0.20:
        penalty += 0.08
    return _clamp(penalty, 0.0, 0.65)


def build_webcam_candidates(samples, width: int, height: int) -> list[WebcamCandidate]:
    total_samples = max(1, len(samples or []))
    clusters = _cluster_face_detections(samples, width, height)
    candidates: list[WebcamCandidate] = []

    for cluster in clusters:
        hits = len(cluster["sample_indices"])
        if hits <= 0:
            continue
        roi = _roi_from_cluster(cluster, width, height)
        persistence = hits / total_samples

        # Presence counts any face whose center repeatedly falls inside the inferred ROI.
        presence_hits = 0
        for sample in samples or []:
            faces = sample[2] if len(sample) >= 3 else []
            found = False
            for x, y, w, h in faces or []:
                cx = float(x) + float(w) / 2.0
                cy = float(y) + float(h) / 2.0
                if roi.x <= cx <= roi.x2 and roi.y <= cy <= roi.y2:
                    found = True
                    break
            presence_hits += int(found)
        presence = presence_hits / total_samples

        stability, geometry = _stability_scores(cluster, width, height)
        visual, temporal = _visual_separation(samples, roi)
        cx = roi.x + roi.w / 2.0
        cy = roi.y + roi.h / 2.0
        edge_score = _near_edge_score(cx, cy, width, height)
        overlay = _clamp(0.42 * edge_score + 0.33 * stability + 0.15 * geometry + 0.10 * visual)
        penalty = _false_positive_penalty(
            cluster, roi, width, height, persistence, stability
        )

        raw_score = (
            0.18 * presence
            + 0.20 * persistence
            + 0.17 * stability
            + 0.10 * geometry
            + 0.14 * overlay
            + 0.11 * visual
            + 0.10 * temporal
        )
        final_score = _clamp(raw_score - penalty)

        candidates.append(
            WebcamCandidate(
                roi=roi,
                corner=_corner_for_center(cx, cy, width, height),
                face_presence_score=presence,
                face_persistence_score=persistence,
                roi_stability_score=stability,
                geometric_consistency_score=geometry,
                overlay_likelihood_score=overlay,
                visual_separation_score=visual,
                temporal_separation_score=temporal,
                false_positive_penalty=penalty,
                final_score=final_score,
                hits=hits,
                total_samples=total_samples,
                sample_indices=sorted(cluster["sample_indices"]),
            )
        )

    candidates.sort(
        key=lambda item: (
            item.final_score,
            item.face_persistence_score,
            item.roi_stability_score,
        ),
        reverse=True,
    )
    return candidates


def detect_webcam(samples, width: int, height: int) -> WebcamDetectionResult:
    candidates = build_webcam_candidates(samples, width, height)
    if not candidates:
        return WebcamDetectionResult(None, [], False, "no_webcam_candidates")

    top = candidates[0]
    if top.final_score < SMARTCROP_WEBCAM_SCORE_THRESHOLD:
        return WebcamDetectionResult(
            None,
            candidates,
            False,
            f"top_score_below_threshold:{top.final_score:.3f}",
        )

    ambiguous = False
    if len(candidates) >= 2:
        margin = top.final_score - candidates[1].final_score
        if margin < SMARTCROP_WEBCAM_AMBIGUITY_MARGIN:
            ambiguous = True
            if (
                top.face_persistence_score < SMARTCROP_WEBCAM_AMBIGUOUS_MIN_PERSISTENCE
                or top.roi_stability_score < SMARTCROP_WEBCAM_AMBIGUOUS_MIN_STABILITY
            ):
                return WebcamDetectionResult(
                    None,
                    candidates,
                    True,
                    "ambiguous_candidates_need_more_temporal_evidence",
                )

    return WebcamDetectionResult(top, candidates, ambiguous, "selected_best_scored_candidate")
