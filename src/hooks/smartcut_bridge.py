from __future__ import annotations

from src.config import SMARTCUT_MINIMUM_SEGMENT_DURATION
from src.logger import info, warning
from src.smart_cut import refine_edit_plan, validate_final_edit_plan


def refine_edit_plan_preserving_hook(
    *,
    video_name: str,
    clip_index: int,
    original_segments: list[dict],
    transcript: list[dict],
    video_duration: float,
    optimized_start: float | None = None,
) -> list[dict]:
    """Run SmartCut while preserving the Hook Optimizer's final START.

    Hook Optimizer has already performed semantic selection + Whisper boundary
    alignment + micro pre-roll. SmartCut remains free to refine END boundaries
    and later extractive segment boundaries, but it must not undo that semantic
    START by running its wider sentence-boundary search again.
    """
    refined = refine_edit_plan(
        video_name=video_name,
        clip_index=clip_index,
        original_segments=original_segments,
        transcript=transcript,
        video_duration=video_duration,
    )
    if optimized_start is None or not refined:
        return refined

    try:
        protected = max(0.0, min(float(optimized_start), float(video_duration)))
        first_index = min(
            range(len(refined)),
            key=lambda index: float(refined[index]["start"]),
        )
        first = dict(refined[first_index])
        first_end = float(first["end"])

        # Never protect a START that would make the first extractive segment too
        # small for SmartCut's own safety constraints.
        if first_end - protected < SMARTCUT_MINIMUM_SEGMENT_DURATION:
            warning(
                "[HOOK->SMARTCUT] Protected START would create a micro-segment; "
                "keeping SmartCut boundary."
            )
            return refined

        before = float(first["start"])
        first["start"] = round(protected, 4)
        candidate = [dict(item) for item in refined]
        candidate[first_index] = first
        candidate.sort(key=lambda item: (float(item["start"]), float(item["end"])))

        if not validate_final_edit_plan(candidate, video_duration):
            warning(
                "[HOOK->SMARTCUT] Protected START failed final plan validation; "
                "keeping SmartCut boundary."
            )
            return refined

        info(
            f"[HOOK->SMARTCUT] Protected optimized START "
            f"{before:.2f} -> {protected:.2f}; SmartCut cannot move semantic hook."
        )
        return candidate
    except (KeyError, TypeError, ValueError) as exc:
        warning(
            f"[HOOK->SMARTCUT] Could not protect optimized START: {exc}; "
            "keeping SmartCut result."
        )
        return refined
