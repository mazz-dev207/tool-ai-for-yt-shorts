from __future__ import annotations

from src.v3.models import VisualEvent

EVENT_TO_EFFECT = {
    "surprise": "punch_in", "major_payoff": "punch_in", "payoff": "punch_in",
    "reaction": "face_zoom", "huge_reaction": "face_zoom",
    "important_detail": "focus_crop", "confusion": "caption_emphasis", "reveal": "punch_in",
}


def _budget_for_intensity(intensity: float) -> int:
    if intensity < 0.40:
        return 1
    if intensity < 0.78:
        return 2
    return 3


def build_visual_events(proposal: dict, *, max_effects_per_event: int = 2, max_total_effects: int = 6, content_profile: str = "auto") -> list[VisualEvent]:
    if content_profile == "podcast":
        max_total_effects = min(max_total_effects, 3)
    result = []
    for raw in proposal.get("visual_events", []) or []:
        if len(result) >= max_total_effects:
            break
        event = str(raw.get("event", "")).lower()
        intensity = max(0.0, min(1.0, float(raw.get("intensity", 0.5) or 0.5)))
        requested = raw.get("edit") or {}
        effects = []
        explicit = str(requested.get("effect", "") or "").lower()
        if explicit:
            effects.append(explicit)
        inferred = EVENT_TO_EFFECT.get(event)
        if inferred and inferred not in effects:
            effects.append(inferred)
        if requested.get("freeze_frame") and "freeze_frame" not in effects:
            effects.append("freeze_frame")
        if requested.get("replay") and "replay" not in effects:
            effects.append("replay")
        budget = min(max_effects_per_event, _budget_for_intensity(intensity))
        for effect in effects[:budget]:
            item = VisualEvent(
                time=float(raw.get("time", 0.0) or 0.0), effect=effect,
                intensity=intensity, duration=float(raw.get("duration", 0.6) or 0.6),
                target=str(raw.get("target", "auto")),
                caption_emphasis=str(requested.get("caption_emphasis", "") or ""),
                metadata={"semantic_event": event},
            )
            if not item.validate():
                result.append(item)
    return result[:max_total_effects]
