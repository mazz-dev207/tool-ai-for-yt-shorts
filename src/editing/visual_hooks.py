from __future__ import annotations

from typing import Any

from src.v3.models import TimelineSegment, VisualEvent


VISUAL_HOOK_TECHNIQUES = {
    "none",
    "camera_whip",
    "show_result_first",
    "unexpected_angle",
    "action_in_motion",
    "object_interaction",
}

_SOURCE_NATIVE_TECHNIQUES = {
    "unexpected_angle",
    "action_in_motion",
    "object_interaction",
}

# Used only when the multimodal proposal does not provide enough visual-hook
# evidence (for example quota/fallback). Keep this deliberately high so the
# tool does not decorate every gaming clip with the same opening effect.
_FALLBACK_GAMING_CAMERA_WHIP_MIN_HOOK_SCORE = 95.0


def _text_blob(proposal: dict) -> str:
    values: list[str] = []
    for item in proposal.get("recommended_transformations", []) or []:
        values.append(str(item))
    analysis = proposal.get("originality_analysis") or {}
    for item in analysis.get("recommended_transformations", []) or []:
        values.append(str(item))
    for raw in proposal.get("visual_events", []) or []:
        values.extend(
            [
                str(raw.get("event", "")),
                str(raw.get("target", "")),
                str((raw.get("edit") or {}).get("effect", "")),
            ]
        )
    return " ".join(values).lower()


def _normalized_explicit(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    technique = str(raw.get("technique", "none") or "none").strip().lower()
    if technique not in VISUAL_HOOK_TECHNIQUES:
        return None
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5) or 0.5)))
    except (TypeError, ValueError):
        confidence = 0.5
    try:
        source_start = float(raw["source_start"]) if raw.get("source_start") is not None else None
        source_end = float(raw["source_end"]) if raw.get("source_end") is not None else None
    except (TypeError, ValueError):
        source_start = source_end = None
    if source_start is not None and source_end is not None and source_end <= source_start:
        source_start = source_end = None
    return {
        "technique": technique,
        "confidence": confidence,
        "source_supported": bool(raw.get("source_supported", False)),
        "source_start": source_start,
        "source_end": source_end,
        "apply_mode": str(raw.get("apply_mode", "none") or "none").strip().lower(),
        "direction": str(raw.get("direction", "left_to_right") or "left_to_right").strip().lower(),
        "reason": str(raw.get("reason", "") or "").strip(),
        "inferred": False,
    }


def _timeline_bounds(timeline: list[TimelineSegment]) -> tuple[float, float] | None:
    if not timeline:
        return None
    return (
        min(float(item.source_start) for item in timeline),
        max(float(item.source_end) for item in timeline),
    )


def _payoff_preview(timeline: list[TimelineSegment]) -> tuple[float, float] | None:
    for item in timeline:
        if item.purpose not in {"payoff", "reaction", "aftermath"}:
            continue
        duration = min(1.20, max(0.45, item.duration))
        return float(item.source_start), min(float(item.source_end), float(item.source_start) + duration)
    return None


def _first_semantic_event(proposal: dict) -> dict | None:
    events = [item for item in proposal.get("visual_events", []) or [] if isinstance(item, dict)]
    if not events:
        return None
    return min(events, key=lambda item: float(item.get("time", 9999.0) or 9999.0))


def _hook_score(hook) -> float:
    try:
        return max(0.0, min(100.0, float(getattr(hook, "score", 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _fallback_camera_whip(hook, content_profile: str) -> dict | None:
    """Conservative no-extra-AI fallback for very strong gaming openings.

    This exists so a temporary Gemini quota/error does not silently disable the
    entire Visual Hook Engine. It only creates the one technique that is a safe
    editorial transform of existing pixels: camera_whip. Source-native actions
    such as object interaction or unusual camera angles are never fabricated.
    """
    if str(content_profile or "").strip().lower() != "gaming":
        return None
    score = _hook_score(hook)
    if score < _FALLBACK_GAMING_CAMERA_WHIP_MIN_HOOK_SCORE:
        return None

    try:
        start = float(getattr(hook, "source_start", 0.0) or 0.0)
    except (TypeError, ValueError):
        start = 0.0
    direction = "left_to_right" if int(round(start * 10.0)) % 2 == 0 else "right_to_left"
    confidence = min(0.78, 0.64 + (score - _FALLBACK_GAMING_CAMERA_WHIP_MIN_HOOK_SCORE) * 0.035)
    return {
        "technique": "camera_whip",
        "confidence": confidence,
        "source_supported": True,
        "source_start": None,
        "source_end": None,
        "apply_mode": "editorial_effect",
        "direction": direction,
        "reason": (
            "Deterministic fallback: very strong upstream gaming Hook score and "
            "no stronger source-grounded visual-hook instruction was available."
        ),
        "inferred": True,
        "fallback": True,
    }


def select_visual_hook(
    proposal: dict,
    *,
    hook,
    timeline: list[TimelineSegment],
    content_profile: str,
) -> dict:
    """Choose at most one opening visual-hook technique.

    The engine never invents physical source actions. Techniques that depend on
    a real camera angle, moving subject or object/lens interaction are only
    labelled source-native unless the multimodal proposal explicitly supplies a
    valid source range. Show-result-first may reuse a real payoff range already
    present in the validated editorial timeline.
    """
    explicit = _normalized_explicit(proposal.get("visual_hook"))
    if explicit is not None:
        return explicit

    text = _text_blob(proposal)
    first_event = _first_semantic_event(proposal)
    existing_first = timeline[0] if timeline else None

    # If editorial reasoning already created a preview/cold-open, classify it
    # instead of stacking a second visual hook on top.
    if existing_first and existing_first.purpose == "cold_open":
        return {
            "technique": "show_result_first",
            "confidence": 0.82,
            "source_supported": True,
            "source_start": float(existing_first.source_start),
            "source_end": float(existing_first.source_end),
            "apply_mode": "timeline_restructure",
            "direction": "none",
            "reason": "Existing validated cold-open preview already leads with a real later source moment.",
            "inferred": True,
        }

    result_first_terms = (
        "show result first", "result first", "payoff first", "open with payoff",
        "lead with payoff", "cold open with payoff", "preview the payoff",
    )
    if any(term in text for term in result_first_terms):
        preview = _payoff_preview(timeline)
        if preview:
            return {
                "technique": "show_result_first",
                "confidence": 0.86,
                "source_supported": True,
                "source_start": preview[0],
                "source_end": preview[1],
                "apply_mode": "timeline_restructure",
                "direction": "none",
                "reason": "Multimodal/editorial recommendation requests a payoff-first opening using a real payoff range.",
                "inferred": True,
            }

    whip_terms = ("camera whip", "whip pan", "fast pan", "rapid pan", "camera spin")
    if any(term in text for term in whip_terms):
        return {
            "technique": "camera_whip",
            "confidence": 0.78,
            "source_supported": True,
            "source_start": None,
            "source_end": None,
            "apply_mode": "editorial_effect",
            "direction": "left_to_right",
            "reason": "Editorial recommendation calls for a rapid opening camera movement.",
            "inferred": True,
        }

    # A strong surprise/reveal inside the first half-second is a safe place for
    # one short editorial whip. It is intentionally disabled for dialogue-first
    # podcast material.
    if first_event and str(content_profile).lower() != "podcast":
        event = str(first_event.get("event", "") or "").lower()
        try:
            event_time = float(first_event.get("time", 999.0) or 999.0)
            intensity = float(first_event.get("intensity", 0.0) or 0.0)
        except (TypeError, ValueError):
            event_time, intensity = 999.0, 0.0
        if event_time <= 0.50 and event in {"surprise", "reveal"} and intensity >= 0.72:
            return {
                "technique": "camera_whip",
                "confidence": 0.72,
                "source_supported": True,
                "source_start": None,
                "source_end": None,
                "apply_mode": "editorial_effect",
                "direction": "right_to_left" if event == "reveal" else "left_to_right",
                "reason": "Strong source-grounded surprise/reveal occurs immediately at the opening.",
                "inferred": True,
            }

    source_native_rules = [
        (
            "action_in_motion",
            ("action in motion", "start mid-action", "start in motion", "open mid-action"),
        ),
        (
            "object_interaction",
            ("object interaction", "across the lens", "over the lens", "grab the object", "slide the object"),
        ),
        (
            "unexpected_angle",
            ("unexpected angle", "weird angle", "low angle", "upside down", "ceiling angle"),
        ),
    ]
    for technique, terms in source_native_rules:
        if any(term in text for term in terms):
            start = float(hook.source_start) if getattr(hook, "source_start", None) is not None else None
            end = float(hook.source_end) if getattr(hook, "source_end", None) is not None else None
            return {
                "technique": technique,
                "confidence": 0.68,
                "source_supported": bool(start is not None and end is not None and end > start),
                "source_start": start,
                "source_end": end,
                "apply_mode": "source_native",
                "direction": "none",
                "reason": "Technique is only accepted when the multimodal/source analysis indicates the physical action exists in the footage.",
                "inferred": True,
            }

    fallback = _fallback_camera_whip(hook, content_profile)
    if fallback is not None:
        return fallback

    return {
        "technique": "none",
        "confidence": 0.0,
        "source_supported": False,
        "source_start": None,
        "source_end": None,
        "apply_mode": "none",
        "direction": "none",
        "reason": "No source-grounded visual-hook technique is clearly justified.",
        "inferred": True,
    }


def apply_visual_hook_timeline(
    timeline: list[TimelineSegment],
    visual_hook: dict,
) -> tuple[list[TimelineSegment], bool]:
    technique = str(visual_hook.get("technique", "none") or "none").lower()
    if technique not in {"show_result_first", *_SOURCE_NATIVE_TECHNIQUES}:
        return list(timeline), False
    if not bool(visual_hook.get("source_supported", False)):
        return list(timeline), False

    start = visual_hook.get("source_start")
    end = visual_hook.get("source_end")
    if start is None or end is None:
        return list(timeline), False
    start, end = float(start), float(end)
    if end <= start:
        return list(timeline), False

    bounds = _timeline_bounds(timeline)
    if bounds is not None and (start < bounds[0] - 0.10 or end > bounds[1] + 0.10):
        return list(timeline), False

    if timeline:
        first = timeline[0]
        if abs(first.source_start - start) < 0.08 and abs(first.source_end - end) < 0.08:
            return list(timeline), False

    purpose = "cold_open"
    note = {
        "show_result_first": "Visual Hook: show a real payoff/result first.",
        "action_in_motion": "Visual Hook: source-native action already in motion.",
        "object_interaction": "Visual Hook: source-native object/lens interaction.",
        "unexpected_angle": "Visual Hook: source-native unexpected camera angle.",
    }[technique]
    updated = list(timeline)
    updated.insert(
        0,
        TimelineSegment(
            source_start=start,
            source_end=end,
            purpose=purpose,
            preserve_audio=True,
            playback_rate=1.0,
            semantic_note=note,
        ),
    )
    return updated, True


def build_visual_hook_event(visual_hook: dict) -> VisualEvent | None:
    technique = str(visual_hook.get("technique", "none") or "none").lower()
    if technique != "camera_whip":
        return None
    if str(visual_hook.get("apply_mode", "")) != "editorial_effect":
        return None
    if float(visual_hook.get("confidence", 0.0) or 0.0) < 0.60:
        return None

    direction = str(visual_hook.get("direction", "left_to_right") or "left_to_right")
    return VisualEvent(
        time=0.0,
        effect="focus_crop",
        intensity=0.82,
        duration=0.34,
        target=direction,
        metadata={
            "semantic_event": "visual_hook",
            "visual_hook_technique": "camera_whip",
            "direction": direction,
            "fallback": bool(visual_hook.get("fallback", False)),
        },
    )
