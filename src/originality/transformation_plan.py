from __future__ import annotations

from src.originality.models import OriginalityAnalysis


def choose_transformations(analysis: OriginalityAnalysis, proposal: dict) -> list[str]:
    if analysis.no_transformation_needed:
        return []
    allowed = {
        "replace_original_intro", "add_context_hook", "compress_setup",
        "reconstruct_hook", "reorder_story", "reaction_punch_in",
        "payoff_punch_in", "focus_important_detail", "add_context_overlay",
        "add_sound_accent", "preserve_reaction", "hard_cut_after_reaction",
    }
    normalized: list[str] = []
    for item in analysis.recommended_transformations:
        value = item.lower().strip().replace(" ", "_").replace("-", "_")
        if value in allowed and value not in normalized:
            normalized.append(value)
    for item in proposal.get("recommended_transformations", []) or []:
        value = str(item).lower().strip().replace(" ", "_").replace("-", "_")
        if value in allowed and value not in normalized:
            normalized.append(value)
    return normalized[:8]
