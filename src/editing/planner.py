from __future__ import annotations

from src.editing.sound_design import build_audio_events
from src.editing.visual_grammar import build_visual_events
from src.v3.models import ContextOverlay, EditPlan, TimelineSegment


def _overlays(proposal: dict, enabled: bool) -> list[ContextOverlay]:
    if not enabled:
        return []
    result = []
    for raw in proposal.get("context_overlays", []) or []:
        overlay = ContextOverlay(
            text=str(raw.get("text", "")), start=float(raw.get("start", 0.1) or 0.1),
            duration=float(raw.get("duration", 1.2) or 1.2), purpose=str(raw.get("purpose", "context")),
        )
        if not overlay.validate():
            result.append(overlay)
    return result[:3]


def build_edit_plan(*, clip_index: int, hook, angle, timeline, proposal: dict, no_transformation_needed: bool, content_profile: str, editorial_profile_name: str, enable_context_overlays: bool, enable_semantic_effects: bool, enable_sound_design: bool, max_effects_per_event: int) -> EditPlan:
    timeline = list(timeline)
    if hook.mode == "RECONSTRUCTED" and hook.source_start is not None and hook.source_end is not None:
        same_first = bool(timeline) and abs(timeline[0].source_start - hook.source_start) < 0.08 and abs(timeline[0].source_end - hook.source_end) < 0.08
        if not same_first:
            timeline.insert(0, TimelineSegment(
                source_start=float(hook.source_start), source_end=float(hook.source_end),
                purpose="cold_open", semantic_note="Reconstructed real source hook.",
            ))

    visual_events = build_visual_events(proposal, max_effects_per_event=max_effects_per_event, content_profile=content_profile) if enable_semantic_effects and not no_transformation_needed else []
    audio_events = build_audio_events(proposal, enabled=enable_sound_design and not no_transformation_needed)
    overlays = _overlays(proposal, enable_context_overlays and not no_transformation_needed)

    return EditPlan(
        clip_index=clip_index, hook=hook, timeline=timeline,
        visual_events=visual_events, audio_events=audio_events, context_overlays=overlays,
        caption_emphasis=[
            {"phrase": str(item.get("phrase", ""))[:80], "emphasis": str(item.get("emphasis", "strong"))}
            for item in proposal.get("caption_emphasis", []) or [] if str(item.get("phrase", "")).strip()
        ][:5],
        editorial_angle=angle.primary_angle, viewer_question=angle.viewer_question,
        stakes=angle.stakes, payoff=angle.payoff,
        no_transformation_needed=no_transformation_needed,
        metadata={"content_profile": content_profile, "editorial_profile": editorial_profile_name, "angle_confidence": angle.confidence},
    )
