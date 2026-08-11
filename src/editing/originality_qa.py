from __future__ import annotations

from src.editing.sound_design import audio_safety_warnings
from src.v3.models import EditPlan, OriginalityQAReport, OriginalityResult


def run_originality_qa(*, plan: EditPlan, originality: OriginalityResult, min_score: int, source_duration: float, max_effects_per_event: int) -> OriginalityQAReport:
    problems = plan.validate(source_duration=source_duration, max_effects_per_event=max_effects_per_event)
    warnings = []
    checks = {
        "timeline_valid": not any("segment" in p or "timeline" in p for p in problems),
        "effect_budget": "effect_budget_exceeded" not in problems,
        "hook_quality": plan.hook.score >= 60 or plan.no_transformation_needed,
        "context_independence": originality.context_independence >= 50,
        "audio_safety": True,
        "payoff_preserved": any(x.purpose == "payoff" for x in plan.timeline),
    }
    audio_warnings = audio_safety_warnings(plan.audio_events)
    if audio_warnings:
        warnings.extend(audio_warnings)
        checks["audio_safety"] = False
    if originality.audio_transformation < 35 and plan.metadata.get("content_profile") == "podcast":
        warnings.append("Low audio transformation is acceptable for dialogue-focused content.")
    if originality.originality_score < min_score:
        problems.append("originality_below_threshold")
    if not checks["payoff_preserved"]:
        warnings.append("No explicit payoff segment was identified; using best valid plan.")
    passed = not problems
    return OriginalityQAReport(
        passed=passed, originality_score=originality.originality_score,
        problems=sorted(set(problems)), warnings=sorted(set(warnings)), checks=checks,
    )
