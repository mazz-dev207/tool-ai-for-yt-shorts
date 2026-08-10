from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.config import HOOK_MICRO_PREROLL_MAX
from src.hooks.scoring import clamp_semantic_start, refine_start_locally
from src.smart_cut import flatten_words


_AUDIO_RATE = 16000


def _inside_word(time_value: float, words) -> bool:
    return any(
        word.start + 0.015 < time_value < word.end - 0.015
        for word in words
    )


def _candidate_times(
    semantic_start: float,
    text_start: float,
    transcript: list[dict],
) -> list[float]:
    lower = max(0.0, semantic_start - max(0.10, float(HOOK_MICRO_PREROLL_MAX)))
    upper = semantic_start + 0.12
    values = [semantic_start, text_start]
    words = flatten_words(transcript)
    for word in words:
        if lower <= word.start <= upper:
            values.append(word.start)
    for segment in transcript or []:
        try:
            value = float(segment.get("start", -1.0))
        except (TypeError, ValueError):
            continue
        if lower <= value <= upper:
            values.append(value)
    return sorted({round(value, 4) for value in values if lower <= value <= upper})


def _extract_audio(video_path: Path, start: float, end: float) -> tuple[np.ndarray, float]:
    if not video_path.exists() or end <= start:
        return np.array([], dtype=np.float32), start
    duration = end - start
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start:.4f}",
        "-t",
        f"{duration:.4f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_AUDIO_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=12,
        )
    except Exception:
        return np.array([], dtype=np.float32), start
    if result.returncode != 0 or not result.stdout:
        return np.array([], dtype=np.float32), start
    raw = np.frombuffer(result.stdout, dtype=np.int16)
    if raw.size == 0:
        return np.array([], dtype=np.float32), start
    return raw.astype(np.float32) / 32768.0, start


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def _audio_boundary_score(
    signal: np.ndarray,
    signal_start: float,
    candidate: float,
) -> float:
    if signal.size == 0:
        return 0.0
    center = int(round((candidate - signal_start) * _AUDIO_RATE))
    pre = int(0.10 * _AUDIO_RATE)
    post = int(0.14 * _AUDIO_RATE)
    if center <= 0 or center >= signal.size:
        return 0.0
    before = signal[max(0, center - pre) : center]
    after = signal[center : min(signal.size, center + post)]
    pre_rms = _rms(before)
    post_rms = _rms(after)

    # Prefer a clean speech/action onset: quieter immediately before, meaningful
    # energy immediately after. Continuous speech remains close to neutral.
    onset = max(0.0, post_rms - pre_rms)
    onset_score = min(1.0, onset / 0.08)
    after_presence = min(1.0, post_rms / 0.055)
    quiet_pre = max(0.0, min(1.0, (0.055 - pre_rms) / 0.055))
    return max(0.0, min(1.0, 0.50 * onset_score + 0.30 * after_presence + 0.20 * quiet_pre))


def _read_gray_frame(capture: cv2.VideoCapture, time_value: float):
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_value) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None, None
    height, width = frame.shape[:2]
    scale = min(1.0, 480.0 / max(1.0, float(width)))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, frame


def _largest_face(frame) -> tuple[float, float, float] | None:
    if frame is None:
        return None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.10,
        minNeighbors=5,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    return x + w / 2.0, y + h / 2.0, float(w * h)


def _visual_and_reaction_score(video_path: Path, candidate: float) -> tuple[float, float]:
    if not video_path.exists():
        return 0.0, 0.0
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0, 0.0
    try:
        before_gray, before_frame = _read_gray_frame(capture, candidate - 0.10)
        current_gray, current_frame = _read_gray_frame(capture, candidate)
        after_gray, after_frame = _read_gray_frame(capture, candidate + 0.10)
    finally:
        capture.release()

    if before_gray is None or current_gray is None or after_gray is None:
        return 0.0, 0.0
    if before_gray.shape != current_gray.shape or current_gray.shape != after_gray.shape:
        return 0.0, 0.0

    cut_delta = float(cv2.absdiff(before_gray, current_gray).mean()) / 255.0
    motion_after = float(cv2.absdiff(current_gray, after_gray).mean()) / 255.0
    shot_boundary = min(1.0, cut_delta / 0.14)
    useful_motion = min(1.0, motion_after / 0.10)
    visual_score = max(0.0, min(1.0, 0.45 * shot_boundary + 0.55 * useful_motion))

    face_before = _largest_face(before_frame)
    face_after = _largest_face(after_frame)
    if face_before is None or face_after is None:
        return visual_score, 0.0

    bx, by, ba = face_before
    ax, ay, aa = face_after
    diagonal = max(1.0, math.hypot(after_frame.shape[1], after_frame.shape[0]))
    movement = math.hypot(ax - bx, ay - by) / diagonal
    area_change = abs(aa - ba) / max(1.0, ba)
    reaction_score = max(
        0.0,
        min(1.0, movement / 0.08 * 0.55 + min(1.0, area_change / 0.35) * 0.45),
    )
    return visual_score, reaction_score


def refine_start_multimodally(
    *,
    video_path: Path,
    semantic_start: float,
    transcript: list[dict],
    original_start: float,
    highlight_end: float,
) -> float:
    """Choose the exact local boundary after Gemini chooses the semantic zone.

    Signals are deliberately weak relative to semantic proximity. This stage may
    polish within the micro-pre-roll neighborhood but cannot search for a new hook.
    It is fail-safe: unavailable FFmpeg/OpenCV/audio simply falls back to the
    existing Whisper/speech-boundary refinement.
    """
    text_start = refine_start_locally(
        semantic_start,
        transcript,
        original_start,
        highlight_end,
    )
    candidates = _candidate_times(semantic_start, text_start, transcript)
    if not candidates:
        return text_start

    words = flatten_words(transcript)
    valid = [value for value in candidates if not _inside_word(value, words)]
    if not valid:
        return text_start

    audio_start = max(0.0, min(valid) - 0.16)
    audio_end = max(valid) + 0.20
    signal, signal_start = _extract_audio(video_path, audio_start, audio_end)

    scored: list[tuple[float, float]] = []
    for candidate in valid:
        proximity = max(0.0, 1.0 - abs(candidate - semantic_start) / 0.45)
        text_proximity = max(0.0, 1.0 - abs(candidate - text_start) / 0.30)
        audio = _audio_boundary_score(signal, signal_start, candidate)
        visual, reaction = _visual_and_reaction_score(video_path, candidate)

        # Semantic/text boundaries dominate. Multimodal features only decide the
        # exact frame inside a very small neighborhood.
        score = (
            1.70 * proximity
            + 1.50 * text_proximity
            + 0.55 * audio
            + 0.35 * visual
            + 0.30 * reaction
        )
        scored.append((score, candidate))

    best = max(scored, key=lambda item: (item[0], -abs(item[1] - text_start)))[1]
    return clamp_semantic_start(best, original_start, highlight_end)
