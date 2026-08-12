from __future__ import annotations

from src.satisfaction.context_independence import analyze_context_independence
from src.satisfaction.expectation_matcher import analyze_expectation_match
from src.satisfaction.models import (
    EndingCandidate,
    ProtectedRange,
    SatisfactionResult,
    SatisfactionRisks,
    SatisfactionStructure,
)
from src.satisfaction.payoff_analyzer import analyze_payoff
from src.satisfaction.scoring import calculate_satisfaction_score
from src.v3.models import EditPlan
from src.v3_config import V3_SATISFACTION_ENABLED


_END_OFFSETS = (-0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
_PREVIEW_PURPOSES = {"cold_open", "replay", "callback"}


def _bounded_score(value) -> int | None:
    if value is None:
        return None
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def build_end_candidate_values(
    clip_end: float,
    context_end: float,
    video_duration: float | None = None,
) -> list[float]:
    upper = float(context_end)
    if video_duration is not None:
        upper = min(upper, float(video_duration))
    values: list[float] = []
    for offset in _END_OFFSETS:
        value = max(0.0, min(upper, float(clip_end) + offset))
        value = round(value, 3)
        if value not in values:
            values.append(value)
    return values


def _structure(raw: dict) -> SatisfactionStructure:
    value = raw.get("structure") if isinstance(raw.get("structure"), dict) else {}
    return SatisfactionStructure(
        setup=bool(value.get("setup", False)),
        tension=bool(value.get("tension", False)),
        escalation=bool(value.get("escalation", False)),
        payoff=bool(value.get("payoff", False)),
        pattern=str(value.get("pattern", "unknown") or "unknown")[:120],
    )


def _risks(raw: dict, clickbait_gap_score: int) -> SatisfactionRisks:
    value = raw.get("risks") if isinstance(raw.get("risks"), dict) else {}
    return SatisfactionRisks(
        confusing_start=bool(value.get("confusing_start", False)),
        missing_context=bool(value.get("missing_context", False)),
        weak_payoff=bool(value.get("weak_payoff", False)),
        clickbait_gap=bool(value.get("clickbait_gap", False)),
        abrupt_ending=bool(value.get("abrupt_ending", False)),
        dead_air=bool(value.get("dead_air", False)),
        generic_outro=bool(value.get("generic_outro", False)),
        clickbait_gap_score=clickbait_gap_score,
    ).normalize()


def _protected_ranges(raw: dict, plan: EditPlan, video_duration: float) -> list[ProtectedRange]:
    result: list[ProtectedRange] = []
    for item in raw.get("protected_ranges", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item.get("start")))
            end = min(float(video_duration), float(item.get("end")))
        except (TypeError, ValueError):
            continue
        reason = str(item.get("reason", "semantic_pause") or "semantic_pause")
        if end > start + 0.03:
            result.append(ProtectedRange(start=start, end=end, reason=reason).normalize())

    # Payoff/reaction beats are protected even if Gemini forgets to emit an
    # explicit protected range. These are already source-grounded by EditPlan.
    for segment in plan.timeline:
        if segment.purpose not in {"payoff", "reaction", "aftermath"}:
            continue
        candidate = ProtectedRange(
            start=float(segment.source_start),
            end=float(segment.source_end),
            reason=f"semantic_{segment.purpose}",
        ).normalize()
        if not any(
            abs(existing.start - candidate.start) < 0.03
            and abs(existing.end - candidate.end) < 0.03
            for existing in result
        ):
            result.append(candidate)
    return result[:10]


def _ending_candidates(raw: dict, allowed_values: list[float]) -> list[EndingCandidate]:
    result: list[EndingCandidate] = []
    for item in raw.get("ending_candidates", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if allowed_values:
            nearest = min(allowed_values, key=lambda value: abs(value - end))
            if abs(nearest - end) > 0.12:
                continue
            end = nearest
        result.append(
            EndingCandidate(
                end=end,
                payoff_score=item.get("payoff_score"),
                emotional_completeness_score=item.get("emotional_completeness_score"),
                ending_quality_score=item.get("ending_quality_score"),
                viewer_satisfaction_score=item.get("viewer_satisfaction_score"),
                reason=str(item.get("reason", "") or ""),
            ).normalize()
        )
    return result[:6]


def _best_ending(candidates: list[EndingCandidate]) -> float | None:
    if not candidates:
        return None

    def score(item: EndingCandidate) -> float:
        values = [
            (item.payoff_score, 0.35),
            (item.emotional_completeness_score, 0.25),
            (item.ending_quality_score, 0.25),
            (item.viewer_satisfaction_score, 0.15),
        ]
        available = [(float(value), weight) for value, weight in values if value is not None]
        if not available:
            return -1.0
        total_weight = sum(weight for _value, weight in available)
        return sum(value * weight for value, weight in available) / total_weight

    winner = max(candidates, key=score)
    return winner.end if score(winner) >= 0 else None


def analyze_viewer_satisfaction(
    *,
    clip: dict,
    plan: EditPlan,
    originality,
    proposal: dict,
    content_profile: str,
    video_duration: float,
    context_end: float,
) -> SatisfactionResult:
    if not V3_SATISFACTION_ENABLED:
        return SatisfactionResult(
            status="unavailable",
            evidence_source="disabled",
            payoff=analyze_payoff(None, plan),
        ).normalize()

    raw = proposal.get("viewer_satisfaction") if isinstance(proposal, dict) else None
    if not isinstance(raw, dict):
        return SatisfactionResult(
            status="unavailable",
            evidence_source="editorial_multimodal_missing",
            payoff=analyze_payoff(None, plan),
            recommended_changes=["Viewer Satisfaction analysis unavailable; existing ranking preserved."],
        ).normalize()

    payoff_score = _bounded_score(raw.get("payoff_score"))
    expectation_score, hook_promise, actual_payoff, clickbait_gap_score = analyze_expectation_match(raw)
    originality_context = getattr(originality, "context_independence", None)
    context_score, context_reason = analyze_context_independence(
        raw,
        originality_context_score=originality_context,
    )
    clarity_score = _bounded_score(raw.get("clarity_score"))
    emotional_score = _bounded_score(raw.get("emotional_completeness_score"))
    value_density_score = _bounded_score(raw.get("value_density_score"))
    ending_score = _bounded_score(raw.get("ending_quality_score"))

    calculated = calculate_satisfaction_score(
        profile=content_profile,
        payoff_score=payoff_score,
        expectation_match_score=expectation_score,
        context_independence_score=context_score,
        clarity_score=clarity_score,
        emotional_completeness_score=emotional_score,
        value_density_score=value_density_score,
        ending_quality_score=ending_score,
    )

    supplied = [
        payoff_score,
        expectation_score,
        context_score,
        clarity_score,
        emotional_score,
        value_density_score,
        ending_score,
    ]
    available_count = sum(value is not None for value in supplied)
    status = "available" if available_count == len(supplied) and calculated is not None else (
        "partial" if calculated is not None else "unavailable"
    )

    score_reasons = raw.get("score_reasons") if isinstance(raw.get("score_reasons"), dict) else {}
    reasons = {
        "payoff": str(score_reasons.get("payoff_reason", "") or ""),
        "expectation_match": str(score_reasons.get("expectation_match_reason", "") or ""),
        "context_independence": context_reason,
        "clarity": str(score_reasons.get("clarity_reason", "") or ""),
        "emotional_completeness": str(score_reasons.get("emotional_completeness_reason", "") or ""),
        "value_density": str(score_reasons.get("value_density_reason", "") or ""),
        "ending_quality": str(score_reasons.get("ending_quality_reason", "") or ""),
    }

    allowed_end_values = build_end_candidate_values(
        float(clip.get("end", 0.0)),
        float(context_end),
        float(video_duration),
    )
    ending_candidates = _ending_candidates(raw, allowed_end_values)
    recommended_end = raw.get("recommended_end")
    try:
        recommended_end = float(recommended_end) if recommended_end is not None else None
    except (TypeError, ValueError):
        recommended_end = None
    if recommended_end is not None and allowed_end_values:
        nearest = min(allowed_end_values, key=lambda value: abs(value - recommended_end))
        recommended_end = nearest if abs(nearest - recommended_end) <= 0.12 else None
    if recommended_end is None:
        recommended_end = _best_ending(ending_candidates)

    result = SatisfactionResult(
        status=status,
        viewer_satisfaction_score=calculated,
        payoff_score=payoff_score,
        expectation_match_score=expectation_score,
        context_independence_score=context_score,
        clarity_score=clarity_score,
        emotional_completeness_score=emotional_score,
        value_density_score=value_density_score,
        ending_quality_score=ending_score,
        structure=_structure(raw),
        payoff=analyze_payoff(raw, plan),
        risks=_risks(raw, clickbait_gap_score),
        reasons=reasons,
        hook_promise=hook_promise,
        actual_payoff=actual_payoff,
        recommended_changes=list(raw.get("recommended_changes", []) or []),
        protected_ranges=_protected_ranges(raw, plan, video_duration),
        ending_candidates=ending_candidates,
        recommended_end=recommended_end,
        evidence_source="gemini_editorial_multimodal",
    )
    return result.normalize()


def _ranges_overlap(start: float, end: float, other_start: float, other_end: float) -> bool:
    return min(end, other_end) - max(start, other_start) > 0.02


def apply_satisfaction_end_optimization(
    *,
    plan: EditPlan,
    satisfaction: SatisfactionResult,
    clip: dict,
    video_duration: float,
    context_end: float,
) -> tuple[bool, str]:
    if not satisfaction.available or satisfaction.recommended_end is None:
        return False, "satisfaction_end_unavailable"

    original_end = float(clip.get("end", 0.0))
    candidate_end = min(
        float(video_duration),
        float(context_end),
        float(satisfaction.recommended_end),
    )
    if candidate_end < original_end - 0.55 or candidate_end > original_end + 3.05:
        return False, "recommended_end_outside_allowed_offsets"

    target_index = None
    for index in range(len(plan.timeline) - 1, -1, -1):
        if plan.timeline[index].purpose not in _PREVIEW_PURPOSES:
            target_index = index
            break
    if target_index is None:
        return False, "no_normal_timeline_segment"

    target = plan.timeline[target_index]
    if abs(candidate_end - target.source_end) <= 0.05:
        return False, "recommended_end_matches_current"
    if candidate_end <= target.source_start + 0.30:
        return False, "recommended_end_would_destroy_final_segment"

    for index, other in enumerate(plan.timeline):
        if index == target_index or other.purpose in _PREVIEW_PURPOSES:
            continue
        if _ranges_overlap(
            float(target.source_start),
            candidate_end,
            float(other.source_start),
            float(other.source_end),
        ):
            # Existing overlap with an earlier segment is tolerated only when it
            # already existed; do not create a new overlap by extending END.
            if not _ranges_overlap(
                float(target.source_start),
                float(target.source_end),
                float(other.source_start),
                float(other.source_end),
            ):
                return False, "recommended_end_creates_source_overlap"

    old_end = float(target.source_end)
    target.source_end = candidate_end
    satisfaction.end_adjustment_applied = True
    satisfaction.end_adjustment_reason = (
        f"ending candidate {old_end:.3f}s -> {candidate_end:.3f}s selected for payoff/completion"
    )
    return True, satisfaction.end_adjustment_reason
