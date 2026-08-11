from __future__ import annotations

from src.editing.sound_design import audio_safety_warnings
from src.v3.models import EditPlan, OriginalityQAReport, OriginalityResult


def run_originality_qa(
    *,
    plan: EditPlan,
    originality: OriginalityResult,
    min_score: int,
    source_duration: float,
    max_effects_per_event: int,
) -> OriginalityQAReport:
    problems = plan.validate(
        source_duration=source_duration,
        max_effects_per_event=max_effects_per_event,
    )
    warnings: list[str] = []
    checks = {
        "timeline_valid": not any(
            "segment" in problem or "timeline" in problem for problem in problems
        ),
        "effect_budget": "effect_budget_exceeded" not in problems,
        "hook_quality": plan.hook.score >= 60 or plan.no_transformation_needed,
        "context_independence": originality.context_independence >= 50,
        "audio_safety": True,
        "payoff_preserved": any(item.purpose == "payoff" for item in plan.timeline),
    }

    audio_warnings = audio_safety_warnings(plan.audio_events)
    if audio_warnings:
        warnings.extend(audio_warnings)
        checks["audio_safety"] = False

    if (
        originality.audio_transformation < 35
        and plan.metadata.get("content_profile") == "podcast"
    ):
        warnings.append(
            "Low audio transformation is acceptable for dialogue-focused content."
        )

    strong_native_exception = (
        plan.no_transformation_needed
        and plan.hook.score >= 80
        and originality.context_independence >= 70
    )
    if originality.originality_score < min_score and not strong_native_exception:
        problems.append("originality_below_threshold")
    elif originality.originality_score < min_score and strong_native_exception:
        warnings.append(
            "No transformation needed: strong standalone source moment kept intentionally simple."
        )
        checks["intentional_simple_edit"] = True

    if not checks["payoff_preserved"]:
        warnings.append(
            "No explicit payoff segment was identified; using best valid plan."
        )

    passed = not problems
    return OriginalityQAReport(
        passed=passed,
        originality_score=originality.originality_score,
        problems=sorted(set(problems)),
        warnings=sorted(set(warnings)),
        checks=checks,
    )
