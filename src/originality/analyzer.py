from __future__ import annotations

from src.originality.models import OriginalityAnalysis


def analyze_originality(
    *,
    proposal: dict,
    clip: dict,
    transcript_text: str,
) -> OriginalityAnalysis:
    # Internal deterministic validators later in the pipeline use the exact
    # transcript context Gemini saw. It is not persisted as public analytics.
    proposal["_truth_context"] = str(transcript_text or "")

    raw = proposal.get("originality_analysis") or {}
    transformations = [
        str(item).strip()
        for item in raw.get("recommended_transformations", []) or []
        if str(item).strip()
    ][:10]
    source_dependency = int(
        max(0, min(100, raw.get("source_dependency", 50) or 50))
    )
    context_independence = int(
        max(0, min(100, raw.get("context_independence", 50) or 50))
    )
    opening_quality = int(
        max(0, min(100, raw.get("current_opening_quality", 50) or 50))
    )
    transformation_need = int(
        max(0, min(100, raw.get("transformation_need", 50) or 50))
    )
    no_transform = bool(raw.get("no_transformation_needed", False))
    if not transformations and opening_quality >= 82 and context_independence >= 78:
        no_transform = True
    return OriginalityAnalysis(
        why_interesting=" ".join(str(raw.get("why_interesting") or "").split()).strip()[:320],
        source_dependency=source_dependency,
        context_independence=context_independence,
        current_opening_quality=opening_quality,
        transformation_need=transformation_need,
        recommended_transformations=transformations,
        no_transformation_needed=no_transform,
    )
