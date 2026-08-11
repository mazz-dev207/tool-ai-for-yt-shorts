from __future__ import annotations

from src.v3.models import EditPlan, OriginalityResult


def calculate_originality_score(*, editorial_transformation: int, narrative_restructuring: int, context_independence: int, visual_transformation: int, hook_originality: int, audio_transformation: int, brand_consistency: int, source_dependency: int) -> int:
    values = {
        "editorial": max(0, min(100, editorial_transformation)),
        "narrative": max(0, min(100, narrative_restructuring)),
        "context": max(0, min(100, context_independence)),
        "visual": max(0, min(100, visual_transformation)),
        "hook": max(0, min(100, hook_originality)),
        "audio": max(0, min(100, audio_transformation)),
        "brand": max(0, min(100, brand_consistency)),
        "dependency": max(0, min(100, source_dependency)),
    }
    score = (
        values["editorial"] * 0.25 + values["narrative"] * 0.20 +
        values["context"] * 0.15 + values["visual"] * 0.15 +
        values["hook"] * 0.10 + values["audio"] * 0.05 +
        values["brand"] * 0.05 + (100 - values["dependency"]) * 0.05
    )
    return max(0, min(100, int(round(score))))


def score_edit_plan(plan: EditPlan, analysis) -> OriginalityResult:
    reordered = any(plan.timeline[index].source_start < plan.timeline[index - 1].source_start for index in range(1, len(plan.timeline)))
    purposes = {item.purpose for item in plan.timeline}
    narrative = 35
    if len(plan.timeline) >= 3:
        narrative += 20
    if {"context", "escalation", "payoff"}.issubset(purposes):
        narrative += 25
    if reordered:
        narrative += 15

    editorial = 35
    editorial += 20 if plan.editorial_angle else 0
    editorial += 15 if plan.context_overlays else 0
    editorial += 15 if plan.hook.mode != "NATIVE" else 0
    editorial += 10 if plan.caption_emphasis else 0

    visual = min(100, 25 + len(plan.visual_events) * 18)
    audio = min(100, 20 + len(plan.audio_events) * 20)
    hook_originality = {"NATIVE": 45, "RECONSTRUCTED": 78, "EDITORIAL": 88}.get(plan.hook.mode, 40)
    if plan.hook.score >= 85:
        hook_originality = min(100, hook_originality + 8)

    brand = 70 if plan.metadata.get("editorial_profile") else 50
    context = int(getattr(analysis, "context_independence", 50))
    dependency = int(getattr(analysis, "source_dependency", 50))
    score = calculate_originality_score(
        editorial_transformation=editorial,
        narrative_restructuring=narrative,
        context_independence=context,
        visual_transformation=visual,
        hook_originality=hook_originality,
        audio_transformation=audio,
        brand_consistency=brand,
        source_dependency=dependency,
    )
    weaknesses: list[str] = []
    if context < 60:
        weaknesses.append("low_context_independence")
    if narrative < 60:
        weaknesses.append("weak_narrative_restructuring")
    if plan.hook.mode == "NATIVE" and plan.hook.score < 75:
        weaknesses.append("weak_opening_transformation")
    if visual < 45 and plan.metadata.get("content_profile") not in {"podcast", "interview"}:
        weaknesses.append("low_visual_transformation")

    return OriginalityResult(
        originality_score=score,
        editorial_transformation=min(100, editorial),
        narrative_restructuring=min(100, narrative),
        context_independence=context,
        visual_transformation=visual,
        hook_originality=hook_originality,
        audio_transformation=audio,
        brand_consistency=brand,
        source_dependency=dependency,
        recommended_transformations=list(getattr(analysis, "recommended_transformations", []) or []),
        weaknesses=weaknesses,
    ).normalize()
