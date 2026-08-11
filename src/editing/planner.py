from __future__ import annotations

from src.editing.sound_design import build_audio_events
from src.editing.visual_grammar import build_visual_events
from src.editing.visual_hooks import (
    apply_visual_hook_timeline,
    build_visual_hook_event,
    select_visual_hook,
)
from src.hooks.v3 import filter_truthful_overlays
from src.v3.models import ContextOverlay, EditPlan, TimelineSegment
from src.v3_config import V3_ENABLE_VISUAL_HOOK_ENGINE, load_editorial_profile


def _overlays(
    proposal: dict,
    enabled: bool,
    editorial_profile: dict,
) -> list[ContextOverlay]:
    overlay_style = str(
        editorial_profile.get("context_overlay_style", "short_high_information")
        or "short_high_information"
    ).lower()
    if not enabled or overlay_style in {"none", "off", "disabled"}:
        return []

    truth_context = str(proposal.get("_truth_context", "") or "")
    raw_items = filter_truthful_overlays(
        proposal.get("context_overlays", []) or [],
        truth_context,
    )
    result = []
    for raw in raw_items:
        overlay = ContextOverlay(
            text=str(raw.get("text", "")),
            start=float(raw.get("start", 0.1) or 0.1),
            duration=float(raw.get("duration", 1.2) or 1.2),
            purpose=str(raw.get("purpose", "context")),
        )
        if not overlay.validate():
            result.append(overlay)
    return result[:3]


def _visual_budget(editorial_profile: dict, configured_max: int) -> int:
    density = str(editorial_profile.get("editing_density", "medium") or "medium").lower()
    if density in {"low", "minimal", "subtle"}:
        return min(1, max(1, configured_max))
    return max(1, configured_max)


def _sound_policy(editorial_profile: dict) -> tuple[bool, int, float]:
    intensity = str(
        editorial_profile.get("sound_design_intensity", "subtle") or "subtle"
    ).lower()
    if intensity in {"none", "off", "disabled"}:
        return False, 0, -18.0
    if intensity in {"subtle", "low"}:
        return True, 2, -12.0
    if intensity in {"high", "strong"}:
        return True, 3, -9.0
    return True, 3, -10.0


def build_edit_plan(
    *,
    clip_index: int,
    hook,
    angle,
    timeline,
    proposal: dict,
    no_transformation_needed: bool,
    content_profile: str,
    editorial_profile_name: str,
    enable_context_overlays: bool,
    enable_semantic_effects: bool,
    enable_sound_design: bool,
    max_effects_per_event: int,
) -> EditPlan:
    editorial_profile = load_editorial_profile()
    timeline = list(timeline)

    if (
        hook.mode == "RECONSTRUCTED"
        and hook.source_start is not None
        and hook.source_end is not None
    ):
        same_first = (
            bool(timeline)
            and abs(timeline[0].source_start - hook.source_start) < 0.08
            and abs(timeline[0].source_end - hook.source_end) < 0.08
        )
        if not same_first:
            timeline.insert(
                0,
                TimelineSegment(
                    source_start=float(hook.source_start),
                    source_end=float(hook.source_end),
                    purpose="cold_open",
                    semantic_note="Reconstructed real source hook.",
                ),
            )

    visual_hook = {
        "technique": "none",
        "confidence": 0.0,
        "source_supported": False,
        "source_start": None,
        "source_end": None,
        "apply_mode": "none",
        "direction": "none",
        "reason": "Visual Hook Engine disabled.",
        "inferred": True,
    }
    visual_hook_timeline_changed = False
    if V3_ENABLE_VISUAL_HOOK_ENGINE:
        visual_hook = select_visual_hook(
            proposal,
            hook=hook,
            timeline=timeline,
            content_profile=content_profile,
        )
        timeline, visual_hook_timeline_changed = apply_visual_hook_timeline(
            timeline,
            visual_hook,
        )

    effective_no_transformation = bool(no_transformation_needed)
    if visual_hook_timeline_changed:
        effective_no_transformation = False

    effect_budget = _visual_budget(
        editorial_profile,
        max_effects_per_event,
    )
    visual_events = (
        build_visual_events(
            proposal,
            max_effects_per_event=effect_budget,
            content_profile=content_profile,
        )
        if enable_semantic_effects and not effective_no_transformation
        else []
    )

    visual_hook_event = (
        build_visual_hook_event(visual_hook)
        if V3_ENABLE_VISUAL_HOOK_ENGINE and enable_semantic_effects
        else None
    )
    if visual_hook_event is not None:
        # One dominant opening hook. Do not stack a second zoom/effect in the
        # same first half-second; later semantic effects remain untouched.
        visual_events = [item for item in visual_events if float(item.time) > 0.55]
        visual_events.insert(0, visual_hook_event)
        effective_no_transformation = False

    profile_sound_enabled, max_audio_events, gain_cap_db = _sound_policy(
        editorial_profile
    )
    audio_events = build_audio_events(
        proposal,
        enabled=(
            enable_sound_design
            and profile_sound_enabled
            and not effective_no_transformation
        ),
        max_total_events=max_audio_events,
        gain_cap_db=gain_cap_db,
    )

    overlays = _overlays(
        proposal,
        enable_context_overlays and not effective_no_transformation,
        editorial_profile,
    )

    return EditPlan(
        clip_index=clip_index,
        hook=hook,
        timeline=timeline,
        visual_events=visual_events,
        audio_events=audio_events,
        context_overlays=overlays,
        caption_emphasis=[
            {
                "phrase": str(item.get("phrase", ""))[:80],
                "emphasis": str(item.get("emphasis", "strong")),
            }
            for item in proposal.get("caption_emphasis", []) or []
            if str(item.get("phrase", "")).strip()
        ][:5],
        editorial_angle=angle.primary_angle,
        viewer_question=angle.viewer_question,
        stakes=angle.stakes,
        payoff=angle.payoff,
        no_transformation_needed=effective_no_transformation,
        metadata={
            "content_profile": content_profile,
            "editorial_profile": str(
                editorial_profile.get("name", editorial_profile_name)
                or editorial_profile_name
            ),
            "editing_density": editorial_profile.get("editing_density", "medium"),
            "sound_design_intensity": editorial_profile.get(
                "sound_design_intensity", "subtle"
            ),
            "ending_behavior": editorial_profile.get(
                "ending_behavior", "hard_cut_after_reaction"
            ),
            "angle_confidence": angle.confidence,
            "visual_hook": visual_hook,
            "visual_hook_timeline_changed": visual_hook_timeline_changed,
        },
    )
