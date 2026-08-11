from __future__ import annotations

from src.v3.models import TimelineSegment


_ALLOWED_PURPOSES = {
    "cold_open", "hook", "context", "setup", "escalation", "payoff",
    "reaction", "aftermath", "callback", "replay",
}


def _fallback_segments(clip: dict) -> list[TimelineSegment]:
    raw = clip.get("segments") or [
        {"start": float(clip["start"]), "end": float(clip["end"]), "role": "context"}
    ]
    result = []
    for index, item in enumerate(raw):
        purpose = str(
            item.get("role") or ("payoff" if index == len(raw) - 1 else "context")
        ).lower()
        if purpose not in _ALLOWED_PURPOSES:
            purpose = "context"
        result.append(
            TimelineSegment(
                source_start=float(item["start"]),
                source_end=float(item["end"]),
                purpose=purpose,
            )
        )
    return result


def restructure_story(
    *,
    clip: dict,
    proposal: dict,
    source_duration: float,
    enable_restructuring: bool = True,
    max_segments: int = 8,
    allowed_start: float | None = None,
    allowed_end: float | None = None,
) -> tuple[list[TimelineSegment], dict]:
    fallback = _fallback_segments(clip)
    if not enable_restructuring:
        return fallback, {"reordered": False, "fallback": "disabled"}
    raw = proposal.get("timeline") or []
    if not raw:
        return fallback, {"reordered": False, "fallback": "no_proposal"}

    segments = []
    ignored_speed_changes = 0
    for item in raw[:max_segments]:
        try:
            requested_rate = float(item.get("playback_rate", 1.0) or 1.0)
            if abs(requested_rate - 1.0) > 0.001:
                ignored_speed_changes += 1
            seg = TimelineSegment(
                source_start=float(item["source_start"]),
                source_end=float(item["source_end"]),
                purpose=str(item.get("purpose", "context")).lower(),
                preserve_audio=bool(item.get("preserve_audio", True)),
                playback_rate=1.0,
                semantic_note=str(item.get("semantic_note", ""))[:240],
            )
        except Exception:
            return fallback, {"reordered": False, "fallback": "invalid_item"}

        if seg.validate(source_duration):
            return fallback, {"reordered": False, "fallback": "invalid_range"}
        if allowed_start is not None and seg.source_start < float(allowed_start) - 0.001:
            return fallback, {"reordered": False, "fallback": "outside_analyzed_context"}
        if allowed_end is not None and seg.source_end > float(allowed_end) + 0.001:
            return fallback, {"reordered": False, "fallback": "outside_analyzed_context"}
        segments.append(seg)

    if not segments:
        return fallback, {"reordered": False, "fallback": "empty"}

    seen = {}
    for seg in segments:
        key = (round(seg.source_start, 2), round(seg.source_end, 2))
        if key in seen and seg.purpose not in {"cold_open", "replay", "callback"}:
            return fallback, {"reordered": False, "fallback": "duplicate_content"}
        seen[key] = seg.purpose

    if sum(seg.duration for seg in segments) < 4.0:
        return fallback, {"reordered": False, "fallback": "timeline_too_short"}

    reordered = any(
        segments[i].source_start < segments[i - 1].source_start
        for i in range(1, len(segments))
    )
    return segments, {
        "reordered": reordered,
        "fallback": None,
        "ignored_speed_changes": ignored_speed_changes,
    }
