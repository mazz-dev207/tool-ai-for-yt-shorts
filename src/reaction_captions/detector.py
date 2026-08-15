from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.config import BASE_DIR
from src.logger import info, warning
from src.reaction_captions.models import ReactionEvent


REACTION_CAPTION_ENABLED = os.getenv("REACTION_CAPTION_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
REACTION_CAPTION_AUDIO_ENABLED = os.getenv(
    "REACTION_CAPTION_AUDIO_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "on"}
REACTION_CAPTION_MODEL = os.getenv(
    "REACTION_CAPTION_MODEL",
    "MIT/ast-finetuned-audioset-10-10-0.4593",
).strip()
REACTION_CAPTION_MIN_SCORE = float(os.getenv("REACTION_CAPTION_MIN_SCORE", "0.34"))
REACTION_CAPTION_WINDOW_SECONDS = max(
    1.0, float(os.getenv("REACTION_CAPTION_WINDOW_SECONDS", "2.5"))
)
REACTION_CAPTION_HOP_SECONDS = max(
    0.35, float(os.getenv("REACTION_CAPTION_HOP_SECONDS", "0.9"))
)
REACTION_CAPTION_MAX_EVENTS = max(
    0, int(os.getenv("REACTION_CAPTION_MAX_EVENTS", "3"))
)
REACTION_CAPTION_MERGE_GAP = max(
    0.0, float(os.getenv("REACTION_CAPTION_MERGE_GAP", "0.8"))
)
REACTION_CAPTION_CACHE_DIR = BASE_DIR / "cache" / "reaction_captions"
_REACTION_CACHE_VERSION = "reaction-caption-audioset-v1"
_AUDIO_RATE = 16000


_KIND_TEXT = {
    "laugh": "[LAUGHS]",
    "gasp": "[GASPS]",
    "scream": "[SCREAMS]",
    "sigh": "[SIGHS]",
    "cry": "[CRYING]",
    "groan": "[GROANS]",
    "cough": "[COUGHS]",
    "sneeze": "[SNEEZES]",
    "applause": "[APPLAUSE]",
    "cheer": "[CHEERING]",
}

_KIND_MIN_SCORE = {
    "laugh": 0.34,
    "gasp": 0.36,
    "scream": 0.42,
    "sigh": 0.36,
    "cry": 0.36,
    "groan": 0.38,
    "cough": 0.40,
    "sneeze": 0.42,
    "applause": 0.40,
    "cheer": 0.40,
}

_LAUGH_LABELS = {
    "laughter",
    "baby laughter",
    "giggle",
    "snicker",
    "belly laugh",
    "chuckle, chortle",
}
_CRY_LABELS = {
    "crying, sobbing",
    "baby cry, infant cry",
    "whimper",
    "wail, moan",
}
_SCREAM_LABELS = {
    "screaming",
    "shout",
    "yell",
    "battle cry",
    "children shouting",
}
_APPLAUSE_LABELS = {"clapping", "applause"}
_CHEER_LABELS = {"cheering", "whoop"}

_classifier = None
_classifier_failed = False
_classifier_warning_emitted = False


def _canonical_kind(label: str) -> str | None:
    value = str(label or "").strip().lower()
    if value in _LAUGH_LABELS:
        return "laugh"
    if value in _CRY_LABELS:
        return "cry"
    if value in _SCREAM_LABELS:
        return "scream"
    if value in _APPLAUSE_LABELS:
        return "applause"
    if value in _CHEER_LABELS:
        return "cheer"
    if value == "gasp":
        return "gasp"
    if value == "sigh":
        return "sigh"
    if value in {"groan", "grunt"}:
        return "groan"
    if value == "cough":
        return "cough"
    if value == "sneeze":
        return "sneeze"
    return None


def _classifier_instance():
    global _classifier, _classifier_failed, _classifier_warning_emitted
    if _classifier is not None:
        return _classifier
    if _classifier_failed:
        return None

    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        info(
            f"[REACTION CAPTION] Încarc AudioSet classifier {REACTION_CAPTION_MODEL} "
            f"pe {'CUDA' if device == 0 else 'CPU'}..."
        )
        _classifier = pipeline(
            "audio-classification",
            model=REACTION_CAPTION_MODEL,
            device=device,
        )
        return _classifier
    except Exception as exc:
        _classifier_failed = True
        if not _classifier_warning_emitted:
            warning(
                "[REACTION CAPTION] classifier audio indisponibil; "
                f"folosesc fallback transcript-only: {exc}"
            )
            _classifier_warning_emitted = True
        return None


def _extract_audio(clip_path: Path) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(clip_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_AUDIO_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
        )
    except Exception as exc:
        raise RuntimeError(f"audio extraction failed: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or "FFmpeg audio extraction failed"
        )
    if not result.stdout:
        return np.array([], dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _audio_windows(signal: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    if signal.size == 0:
        return []
    total_seconds = signal.size / _AUDIO_RATE
    window_samples = max(1, int(round(REACTION_CAPTION_WINDOW_SECONDS * _AUDIO_RATE)))
    hop_samples = max(1, int(round(REACTION_CAPTION_HOP_SECONDS * _AUDIO_RATE)))
    result: list[tuple[float, float, np.ndarray]] = []
    offset = 0
    while offset < signal.size:
        end = min(signal.size, offset + window_samples)
        chunk = signal[offset:end]
        if chunk.size < int(0.45 * _AUDIO_RATE):
            break
        start_time = offset / _AUDIO_RATE
        end_time = min(total_seconds, end / _AUDIO_RATE)
        result.append((start_time, end_time, chunk))
        if end >= signal.size:
            break
        offset += hop_samples
    return result


def _classification_event(
    *,
    start: float,
    end: float,
    label: str,
    score: float,
) -> ReactionEvent | None:
    kind = _canonical_kind(label)
    if kind is None:
        return None
    threshold = max(REACTION_CAPTION_MIN_SCORE, _KIND_MIN_SCORE.get(kind, 0.0))
    if float(score) < threshold:
        return None

    center = (float(start) + float(end)) / 2.0
    duration = min(1.65, max(0.85, float(end) - float(start)))
    event_start = max(float(start), center - duration / 2.0)
    event_end = min(float(end), event_start + duration)
    if event_end - event_start < 0.55:
        event_end = min(float(end), event_start + 0.55)

    return ReactionEvent(
        kind=kind,
        text=_KIND_TEXT[kind],
        start=round(event_start, 3),
        end=round(event_end, 3),
        score=round(float(score), 4),
        source="audio",
        raw_label=str(label),
    )


def _detect_audio_events(clip_path: Path) -> list[ReactionEvent]:
    if not REACTION_CAPTION_AUDIO_ENABLED:
        return []
    classifier = _classifier_instance()
    if classifier is None:
        return []

    signal = _extract_audio(clip_path)
    windows = _audio_windows(signal)
    if not windows:
        return []

    inputs = [
        {"array": chunk, "sampling_rate": _AUDIO_RATE}
        for _, _, chunk in windows
    ]
    try:
        predictions = classifier(
            inputs,
            top_k=12,
            batch_size=min(8, len(inputs)),
        )
    except TypeError:
        predictions = [classifier(item, top_k=12) for item in inputs]

    if predictions and isinstance(predictions[0], dict):
        predictions = [predictions]

    events: list[ReactionEvent] = []
    for (start, end, _chunk), prediction in zip(windows, predictions):
        best: ReactionEvent | None = None
        for item in prediction or []:
            if not isinstance(item, dict):
                continue
            event = _classification_event(
                start=start,
                end=end,
                label=str(item.get("label", "")),
                score=float(item.get("score", 0.0) or 0.0),
            )
            if event is not None and (best is None or event.score > best.score):
                best = event
        if best is not None:
            events.append(best)
    return events


def _iter_group_words(groups) -> list[tuple[str, float, float]]:
    items: list[tuple[str, float, float]] = []
    for group in groups or []:
        for word in getattr(group, "words", []) or []:
            try:
                items.append(
                    (
                        str(getattr(word, "word", "") or ""),
                        float(getattr(word, "start", 0.0) or 0.0),
                        float(getattr(word, "end", 0.0) or 0.0),
                    )
                )
            except (TypeError, ValueError):
                continue
    return items


def _annotation_kind(raw: str) -> str | None:
    text = str(raw or "").strip()
    upper = text.upper()
    compact = re.sub(r"[^A-Z]", "", upper)

    # Repeated ha/hehe is useful even when Whisper did not emit brackets.
    if re.fullmatch(r"(?:HA){2,}|(?:HE){2,}|HAHA+|HEHE+", compact):
        return "laugh"

    is_annotation = (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
        or (text.startswith("<") and text.endswith(">"))
    )
    if not is_annotation:
        return None

    checks = (
        ("laugh", ("LAUGH", "GIGGLE", "CHUCKLE", "SNICKER")),
        ("gasp", ("GASP",)),
        ("scream", ("SCREAM", "YELL", "SHOUT")),
        ("sigh", ("SIGH",)),
        ("cry", ("CRY", "SOB", "WHIMPER", "WAIL")),
        ("groan", ("GROAN", "GRUNT")),
        ("cough", ("COUGH",)),
        ("sneeze", ("SNEEZE",)),
        ("applause", ("APPLAUSE", "CLAP")),
        ("cheer", ("CHEER", "WHOOP")),
    )
    for kind, needles in checks:
        if any(needle in upper for needle in needles):
            return kind
    return None


def _detect_transcript_events(groups) -> list[ReactionEvent]:
    events: list[ReactionEvent] = []
    for raw, start, end in _iter_group_words(groups):
        kind = _annotation_kind(raw)
        if kind is None:
            continue
        event_start = max(0.0, start - 0.12)
        event_end = max(event_start + 0.75, end + 0.40)
        events.append(
            ReactionEvent(
                kind=kind,
                text=_KIND_TEXT[kind],
                start=round(event_start, 3),
                end=round(event_end, 3),
                score=0.92,
                source="transcript",
                raw_label=raw,
            )
        )
    return events


def _overlap(first: ReactionEvent, second: ReactionEvent) -> float:
    return max(0.0, min(first.end, second.end) - max(first.start, second.start))


def _merge_same_kind(events: list[ReactionEvent]) -> list[ReactionEvent]:
    if not events:
        return []
    ordered = sorted(events, key=lambda item: (item.start, item.end, -item.score))
    merged: list[ReactionEvent] = []
    for event in ordered:
        if (
            merged
            and merged[-1].kind == event.kind
            and event.start <= merged[-1].end + REACTION_CAPTION_MERGE_GAP
        ):
            previous = merged[-1]
            previous.end = round(max(previous.end, event.end), 3)
            previous.score = round(max(previous.score, event.score), 4)
            if previous.source != event.source:
                previous.source = "audio+transcript"
            if event.score >= previous.score and event.raw_label:
                previous.raw_label = event.raw_label
            continue
        merged.append(event)
    return merged


def _resolve_conflicts(events: list[ReactionEvent]) -> list[ReactionEvent]:
    selected: list[ReactionEvent] = []
    for event in sorted(events, key=lambda item: (-item.score, item.start)):
        conflict = next(
            (
                existing
                for existing in selected
                if existing.kind != event.kind
                and _overlap(existing, event) >= 0.45
            ),
            None,
        )
        if conflict is None:
            selected.append(event)
    return sorted(selected, key=lambda item: item.start)


def _cap_events(events: list[ReactionEvent]) -> list[ReactionEvent]:
    if REACTION_CAPTION_MAX_EVENTS <= 0:
        return []
    if len(events) <= REACTION_CAPTION_MAX_EVENTS:
        return events
    strongest = sorted(
        events,
        key=lambda item: (item.score, item.duration),
        reverse=True,
    )[:REACTION_CAPTION_MAX_EVENTS]
    return sorted(strongest, key=lambda item: item.start)


def _groups_signature(groups) -> list[list[Any]]:
    return [
        [raw, round(start, 3), round(end, 3)]
        for raw, start, end in _iter_group_words(groups)
    ]


def _cache_key(clip_path: Path, groups) -> str:
    try:
        stat = clip_path.stat()
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
    except OSError:
        size = 0
        mtime_ns = 0
    payload = {
        "path": str(clip_path.resolve()).lower(),
        "size": size,
        "mtime_ns": mtime_ns,
        "model": REACTION_CAPTION_MODEL,
        "audio_enabled": REACTION_CAPTION_AUDIO_ENABLED,
        "min_score": REACTION_CAPTION_MIN_SCORE,
        "window": REACTION_CAPTION_WINDOW_SECONDS,
        "hop": REACTION_CAPTION_HOP_SECONDS,
        "max_events": REACTION_CAPTION_MAX_EVENTS,
        "merge_gap": REACTION_CAPTION_MERGE_GAP,
        "groups": _groups_signature(groups),
        "version": _REACTION_CACHE_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_cache(key: str) -> list[ReactionEvent] | None:
    path = REACTION_CAPTION_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return None
        return [ReactionEvent.from_dict(item) for item in payload if isinstance(item, dict)]
    except Exception:
        return None


def _write_cache(key: str, events: list[ReactionEvent]) -> None:
    try:
        REACTION_CAPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (REACTION_CAPTION_CACHE_DIR / f"{key}.json").write_text(
            json.dumps([item.to_dict() for item in events], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        warning(f"[REACTION CAPTION] cache write failed: {exc}")


def detect_reactions(clip_path: Path, groups=None) -> list[ReactionEvent]:
    """Detect a small number of high-confidence human reaction captions.

    AudioSet/AST runs only on the already-cut Short, never on the entire source video.
    Transcript annotations/repeated laughter provide a zero-dependency fallback.
    """
    if not REACTION_CAPTION_ENABLED:
        return []
    clip_path = Path(clip_path)
    if not clip_path.exists():
        warning(f"[REACTION CAPTION] clip missing: {clip_path}")
        return _detect_transcript_events(groups)

    key = _cache_key(clip_path, groups)
    cached = _read_cache(key)
    if cached is not None:
        info(
            f"[REACTION CAPTION] {clip_path.name} cache hit events={len(cached)}"
        )
        return cached

    transcript_events = _detect_transcript_events(groups)
    audio_events: list[ReactionEvent] = []
    if REACTION_CAPTION_AUDIO_ENABLED:
        try:
            audio_events = _detect_audio_events(clip_path)
        except Exception as exc:
            warning(
                f"[REACTION CAPTION] audio detection failed for {clip_path.name}: {exc}; "
                "continuing with transcript fallback."
            )

    events = _merge_same_kind(transcript_events + audio_events)
    events = _resolve_conflicts(events)
    events = _cap_events(events)
    _write_cache(key, events)

    if events:
        rendered = ", ".join(
            f"{item.text}@{item.start:.2f}s({item.score:.2f})"
            for item in events
        )
        info(f"[REACTION CAPTION] {clip_path.name}: {rendered}")
    else:
        info(f"[REACTION CAPTION] {clip_path.name}: no strong reactions")
    return events
