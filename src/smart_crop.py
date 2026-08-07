from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT

SAMPLE_INTERVAL = 0.50

# Detectăm pe o imagine redusă pentru viteză.
DETECTION_MAX_WIDTH = 640

# Stabilizare.
SMOOTHING_ALPHA = 0.32
DEAD_ZONE_RATIO = 0.07
MAX_MOVE_RATIO = 0.18


# --------------------------------------------------
# DATA
# --------------------------------------------------

@dataclass
class CropPoint:
    time: float
    center_x: float
    center_y: float
    source: str = "fallback"


@dataclass
class CropPlan:
    input_width: int
    input_height: int
    crop_width: int
    crop_height: int
    duration: float
    points: List[CropPoint]


# --------------------------------------------------
# DETECTORS
# --------------------------------------------------

def _load_face_detector():
    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(
        str(cascade_path)
    )

    if detector.empty():
        return None

    return detector


def _load_person_detector():
    hog = cv2.HOGDescriptor()

    hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )

    return hog


FACE_DETECTOR = _load_face_detector()
PERSON_DETECTOR = _load_person_detector()


# --------------------------------------------------
# VIDEO HELPERS
# --------------------------------------------------

def _even(value: float) -> int:
    result = max(2, int(round(value)))

    if result % 2 != 0:
        result -= 1

    return max(2, result)


def _video_info(capture) -> Tuple[int, int, float, float]:
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    frame_count = float(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    duration = (
        frame_count / fps
        if frame_count > 0
        else 0.0
    )

    return width, height, fps, duration


def calculate_crop_size(
    width: int,
    height: int
) -> Tuple[int, int]:
    """
    Calculează cea mai mare zonă 9:16
    care încape în cadrul original.
    """

    source_ratio = width / height

    if source_ratio >= TARGET_RATIO:

        crop_height = height

        crop_width = _even(
            height * TARGET_RATIO
        )

    else:

        crop_width = width

        crop_height = _even(
            width / TARGET_RATIO
        )

    crop_width = min(
        crop_width,
        width
    )

    crop_height = min(
        crop_height,
        height
    )

    return crop_width, crop_height


# --------------------------------------------------
# FRAME PREP
# --------------------------------------------------

def _resize_for_detection(frame):
    height, width = frame.shape[:2]

    if width <= DETECTION_MAX_WIDTH:
        return frame, 1.0

    scale = (
        DETECTION_MAX_WIDTH
        / float(width)
    )

    resized = cv2.resize(
        frame,
        (
            int(width * scale),
            int(height * scale)
        ),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def _choose_candidate(
    candidates,
    previous_center: Optional[Tuple[float, float]]
):
    """
    Dacă avem tracking anterior, preferăm obiectul
    apropiat de poziția precedentă. Altfel alegem
    cea mai mare detecție.
    """

    if not candidates:
        return None

    if previous_center is None:
        return max(
            candidates,
            key=lambda item: item[2] * item[3]
        )

    previous_x, previous_y = previous_center

    def candidate_score(item):
        x, y, w, h = item

        center_x = x + w / 2
        center_y = y + h / 2

        distance = (
            (center_x - previous_x) ** 2
            +
            (center_y - previous_y) ** 2
        ) ** 0.5

        # Candidatul mare primește un mic avantaj.
        area_bonus = (
            w * h
        ) ** 0.5 * 0.20

        return distance - area_bonus

    return min(
        candidates,
        key=candidate_score
    )


# --------------------------------------------------
# FACE DETECTION
# --------------------------------------------------

def detect_face_center(
    frame,
    previous_center: Optional[Tuple[float, float]] = None
):
    if FACE_DETECTOR is None:
        return None

    small, scale = _resize_for_detection(
        frame
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.10,
        minNeighbors=5,
        minSize=(36, 36)
    )

    if len(faces) == 0:
        return None

    previous_small = None

    if previous_center is not None:
        previous_small = (
            previous_center[0] * scale,
            previous_center[1] * scale
        )

    selected = _choose_candidate(
        list(faces),
        previous_small
    )

    if selected is None:
        return None

    x, y, w, h = selected

    center_x = (
        x + w / 2
    ) / scale

    # Punem centrul puțin sub centrul feței.
    # Ajută compoziția să includă și partea superioară
    # a corpului în crop-ul vertical.
    center_y = (
        y + h * 0.72
    ) / scale

    return center_x, center_y


# --------------------------------------------------
# PERSON FALLBACK
# --------------------------------------------------

def detect_person_center(
    frame,
    previous_center: Optional[Tuple[float, float]] = None
):
    if PERSON_DETECTOR is None:
        return None

    small, scale = _resize_for_detection(
        frame
    )

    boxes, _weights = PERSON_DETECTOR.detectMultiScale(
        small,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    if len(boxes) == 0:
        return None

    previous_small = None

    if previous_center is not None:
        previous_small = (
            previous_center[0] * scale,
            previous_center[1] * scale
        )

    selected = _choose_candidate(
        list(boxes),
        previous_small
    )

    if selected is None:
        return None

    x, y, w, h = selected

    center_x = (
        x + w / 2
    ) / scale

    # Pentru persoană, centrul vizual util este
    # ușor deasupra centrului bounding-box-ului.
    center_y = (
        y + h * 0.42
    ) / scale

    return center_x, center_y


# --------------------------------------------------
# STABILIZATION
# --------------------------------------------------

def _clamp(
    value: float,
    minimum: float,
    maximum: float
) -> float:
    return max(
        minimum,
        min(value, maximum)
    )


def _stabilize_axis(
    previous: float,
    target: float,
    visible_size: float,
) -> float:
    dead_zone = (
        visible_size
        * DEAD_ZONE_RATIO
    )

    delta = target - previous

    if abs(delta) <= dead_zone:
        return previous

    max_move = (
        visible_size
        * MAX_MOVE_RATIO
    )

    delta = _clamp(
        delta,
        -max_move,
        max_move
    )

    return (
        previous
        +
        delta * SMOOTHING_ALPHA
    )


def stabilize_points(
    points: List[CropPoint],
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> List[CropPoint]:

    if not points:
        return points

    result = [
        points[0]
    ]

    for point in points[1:]:

        previous = result[-1]

        center_x = _stabilize_axis(
            previous.center_x,
            point.center_x,
            crop_width
        )

        center_y = _stabilize_axis(
            previous.center_y,
            point.center_y,
            crop_height
        )

        center_x = _clamp(
            center_x,
            crop_width / 2,
            frame_width - crop_width / 2
        )

        center_y = _clamp(
            center_y,
            crop_height / 2,
            frame_height - crop_height / 2
        )

        result.append(
            CropPoint(
                time=point.time,
                center_x=center_x,
                center_y=center_y,
                source=point.source
            )
        )

    return result


# --------------------------------------------------
# ANALYZE VIDEO
# --------------------------------------------------

def analyze_smart_crop(
    video_path: Path,
    sample_interval: float = SAMPLE_INTERVAL,
) -> CropPlan:

    video_path = Path(
        video_path
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Nu pot deschide video: {video_path}"
        )

    try:

        width, height, fps, duration = _video_info(
            capture
        )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Dimensiunile video nu au putut fi citite."
            )

        crop_width, crop_height = calculate_crop_size(
            width,
            height
        )

        points: List[CropPoint] = []

        previous_detection = None

        time_position = 0.0

        # Dacă durata nu este disponibilă, analizăm
        # secvențial folosind frame_count.
        if duration <= 0:
            duration = 1.0

        while time_position <= duration + 0.001:

            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                time_position * 1000.0
            )

            ok, frame = capture.read()

            if not ok:
                break

            detected = detect_face_center(
                frame,
                previous_detection
            )

            source = "face"

            if detected is None:

                detected = detect_person_center(
                    frame,
                    previous_detection
                )

                source = "person"

            if detected is None:

                if previous_detection is not None:
                    detected = previous_detection
                    source = "previous"

                else:
                    detected = (
                        width / 2,
                        height / 2
                    )
                    source = "center"

            center_x, center_y = detected

            center_x = _clamp(
                center_x,
                crop_width / 2,
                width - crop_width / 2
            )

            center_y = _clamp(
                center_y,
                crop_height / 2,
                height - crop_height / 2
            )

            points.append(
                CropPoint(
                    time=time_position,
                    center_x=center_x,
                    center_y=center_y,
                    source=source
                )
            )

            previous_detection = (
                center_x,
                center_y
            )

            time_position += sample_interval

        if not points:
            points.append(
                CropPoint(
                    time=0.0,
                    center_x=width / 2,
                    center_y=height / 2,
                    source="center"
                )
            )

        # Ultimul punct trebuie să ajungă până la capătul clipului.
        if points[-1].time < duration:

            last = points[-1]

            points.append(
                CropPoint(
                    time=duration,
                    center_x=last.center_x,
                    center_y=last.center_y,
                    source=last.source
                )
            )

        points = stabilize_points(
            points,
            crop_width,
            crop_height,
            width,
            height
        )

        return CropPlan(
            input_width=width,
            input_height=height,
            crop_width=crop_width,
            crop_height=crop_height,
            duration=duration,
            points=points
        )

    finally:

        capture.release()


# --------------------------------------------------
# FFMPEG SENDCMD
# --------------------------------------------------

def _crop_xy(
    point: CropPoint,
    plan: CropPlan
) -> Tuple[float, float]:

    x = (
        point.center_x
        -
        plan.crop_width / 2
    )

    y = (
        point.center_y
        -
        plan.crop_height / 2
    )

    max_x = (
        plan.input_width
        -
        plan.crop_width
    )

    max_y = (
        plan.input_height
        -
        plan.crop_height
    )

    x = _clamp(
        x,
        0,
        max_x
    )

    y = _clamp(
        y,
        0,
        max_y
    )

    return x, y


def write_sendcmd(
    plan: CropPlan,
    output_file: Path,
    target_name: str = "smart",
) -> Path:
    """
    Scrie comenzi FFmpeg interpolate.

    Exemplu conceptual:
      0-0.5 crop@smart x lerp(...)
    """

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    lines = []

    if len(plan.points) == 1:

        x, y = _crop_xy(
            plan.points[0],
            plan
        )

        lines.append(
            f"0.000 crop@{target_name} x {x:.3f};"
        )

        lines.append(
            f"0.000 crop@{target_name} y {y:.3f};"
        )

    else:

        for index in range(
            len(plan.points) - 1
        ):

            current = plan.points[index]
            nxt = plan.points[index + 1]

            start = current.time
            end = max(
                nxt.time,
                start + 0.001
            )

            x1, y1 = _crop_xy(
                current,
                plan
            )

            x2, y2 = _crop_xy(
                nxt,
                plan
            )

            lines.append(
                f"{start:.3f}-{end:.3f} "
                f"[expr] crop@{target_name} x "
                f"'lerp({x1:.3f},{x2:.3f},TI)';"
            )

            lines.append(
                f"{start:.3f}-{end:.3f} "
                f"[expr] crop@{target_name} y "
                f"'lerp({y1:.3f},{y2:.3f},TI)';"
            )

    output_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return output_file


def initial_crop_xy(
    plan: CropPlan
) -> Tuple[int, int]:

    x, y = _crop_xy(
        plan.points[0],
        plan
    )

    return int(round(x)), int(round(y))
