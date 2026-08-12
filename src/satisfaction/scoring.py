from __future__ import annotations

from src.satisfaction.models import FinalScoreResult, QualityGateResult, SatisfactionResult
from src.v3_config import (
    V3_FINAL_SCORE_WEIGHTS,
    V3_MAX_CLICKBAIT_GAP,
    V3_MIN_CONTEXT_SCORE,
    V3_MIN_PAYOFF_SCORE,
    V3_MIN_SATISFACTION_SCORE,
    V3_SATISFACTION_FALLBACK_TOP_K,
)


_BASE_SATISFACTION_WEIGHTS = {
    "payoff": 0.22,
    "expectation": 0.18,
    "context": 0.14,
    "clarity": 0.12,
    "emotional": 0.14,
    "value_density": 0.10,
    "ending": 0.10,
}

_PROFILE_SATISFACTION_WEIGHTS = {
    "gaming": {
        "payoff": 0.24,
        "expectation": 0.14,
        "context": 0.10,
        "clarity": 0.10,
        "emotional": 0.18,
        "value_density": 0.10,
        "ending": 0.14,
    },
    "podcast": {
        "payoff": 0.20,
        "expectation": 0.20,
        "context": 0.16,
        "clarity": 0.16,
        "emotional": 0.08,
        "value_density": 0.10,
        "ending": 0.10,
    },
    "interview": {
        "payoff": 0.20,
        "expectation": 0.20,
        "context": 0.16,
        "clarity": 0.16,
        "emotional": 0.08,
        "value_density": 0.10,
        "ending": 0.10,
    },
    "entertainment": {
        "payoff": 0.24,
        "expectation": 0.16,
        "context": 0.12,
        "clarity": 0.10,
        "emotional": 0.16,
        "value_density": 0.10,
        "ending": 0.12,
    },
    "reaction": {
        "payoff": 0.22,
        "expectation": 0.16,
        "context": 0.12,
        "clarity": 0.10,
        "emotional": 0.20,
        "value_density": 0.08,
        "ending": 0.12,
    },
    "story": {
        "payoff": 0.22,
        "expectation": 0.18,
        "context": 0.14,
        "clarity": 0.10,
        "emotional": 0.14,
        "value_density": 0.08,
        "ending": 0.14,
    },
}


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return None


def calculate_satisfaction_score(
    *,
    profile: str,
    payoff_score: int | None,
    expectation_match_score: int | None,
    context_independence_score: int | None,
    clarity_score: int | None,
    emotional_completeness_score: int | None,
    value_density_score: int | None,
    ending_quality_score: int | None,
) -> int | None:
    values = {
        "payoff": _number(payoff_score),
        "expectation": _number(expectation_match_score),
        "context": _number(context_independence_score),
        "clarity": _number(clarity_score),
        "emotional": _number(emotional_completeness_score),
        "value_density": _number(value_density_score),
        "ending": _number(ending_quality_score),
    }
    available = {key: value for key, value in values.items() if value is not None}
    if len(available) < 4:
        return None

    weights = _PROFILE_SATISFACTION_WEIGHTS.get(
        str(profile or "general").strip().lower(),
        _BASE_SATISFACTION_WEIGHTS,
    )
    total_weight = sum(weights[key] for key in available)
    if total_weight <= 0:
        return None
    score = sum(available[key] * weights[key] for key in available) / total_weight
    return max(0, min(100, int(round(score))))


def evaluate_quality_gates(result: SatisfactionResult) -> QualityGateResult:
    if not result.available:
        return QualityGateResult(
            status="UNAVAILABLE_FALLBACK",
            passed=True,
            reasons=["viewer_satisfaction_unavailable"],
            needs_review=False,
        )

    reasons: list[str] = []
    if result.viewer_satisfaction_score is not None and result.viewer_satisfaction_score < V3_MIN_SATISFACTION_SCORE:
        reasons.append(
            f"satisfaction {result.viewer_satisfaction_score} < {V3_MIN_SATISFACTION_SCORE}"
        )
    if result.payoff_score is not None and result.payoff_score < V3_MIN_PAYOFF_SCORE:
        reasons.append(f"payoff {result.payoff_score} < {V3_MIN_PAYOFF_SCORE}")
    if result.context_independence_score is not None and result.context_independence_score < V3_MIN_CONTEXT_SCORE:
        reasons.append(
            f"context {result.context_independence_score} < {V3_MIN_CONTEXT_SCORE}"
        )
    if result.risks.clickbait_gap_score > V3_MAX_CLICKBAIT_GAP:
        reasons.append(
            f"clickbait_gap {result.risks.clickbait_gap_score} > {V3_MAX_CLICKBAIT_GAP}"
        )

    return QualityGateResult(
        status="PASS" if not reasons else "REJECT",
        passed=not reasons,
        reasons=reasons,
        needs_review=bool(reasons),
    )


def calculate_final_short_score(
    *,
    hook_score: float | int | None,
    retention_score: float | int | None,
    originality_score: float | int | None,
    satisfaction: SatisfactionResult,
) -> FinalScoreResult:
    components = {
        "hook": _number(hook_score),
        "retention": _number(retention_score),
        "originality": _number(originality_score),
        "satisfaction": _number(satisfaction.viewer_satisfaction_score) if satisfaction.available else None,
        "payoff": _number(satisfaction.payoff_score) if satisfaction.available else None,
        "context": _number(satisfaction.context_independence_score) if satisfaction.available else None,
    }
    available = {
        key: value
        for key, value in components.items()
        if value is not None and float(V3_FINAL_SCORE_WEIGHTS.get(key, 0.0)) > 0
    }
    unavailable = [key for key, value in components.items() if value is None]
    if not available:
        return FinalScoreResult(
            score=None,
            components={},
            effective_weights={},
            unavailable_components=unavailable,
        )

    raw_weight_sum = sum(float(V3_FINAL_SCORE_WEIGHTS[key]) for key in available)
    if raw_weight_sum <= 0:
        return FinalScoreResult(
            score=None,
            components=available,
            effective_weights={},
            unavailable_components=unavailable,
        )

    effective = {
        key: float(V3_FINAL_SCORE_WEIGHTS[key]) / raw_weight_sum
        for key in available
    }
    score = sum(available[key] * effective[key] for key in available)
    return FinalScoreResult(
        score=round(max(0.0, min(100.0, score)), 2),
        components={key: round(value, 2) for key, value in available.items()},
        effective_weights={key: round(value, 5) for key, value in effective.items()},
        unavailable_components=unavailable,
    )


def _record_score(record: dict) -> tuple[float, float, int]:
    final_score = record.get("final_score")
    score = getattr(final_score, "score", None)
    if score is None and isinstance(final_score, dict):
        score = final_score.get("score")
    try:
        parsed = float(score)
    except (TypeError, ValueError):
        parsed = -1.0
    clip = record.get("clip") or {}
    try:
        retention = float(clip.get("retention_score", 0) or 0)
    except (TypeError, ValueError):
        retention = 0.0
    return parsed, retention, -int(record.get("original_index", 0) or 0)


def rank_satisfaction_records(records: list[dict]) -> tuple[list[dict], list[dict], str]:
    """Apply gates without turning a temporary AI outage into a destructive filter."""
    if not records:
        return [], [], "empty"

    satisfaction_available = [
        record for record in records
        if getattr(record.get("satisfaction"), "available", False)
    ]
    if not satisfaction_available:
        # Preserve existing Retention/Hook order exactly when Satisfaction is unavailable.
        return list(records), [], "satisfaction_unavailable_preserve_order"

    passed = [
        record for record in records
        if getattr(record.get("gate"), "status", "") == "PASS"
    ]
    unavailable = [
        record for record in records
        if getattr(record.get("gate"), "status", "") == "UNAVAILABLE_FALLBACK"
    ]
    rejected = [
        record for record in records
        if getattr(record.get("gate"), "status", "") == "REJECT"
    ]

    # Partial API failure creates incomparable score coverage. Filter only the
    # candidates with known hard-gate failures, then preserve the upstream order
    # for all eligible PASS/unavailable records instead of ranking apples vs pears.
    if unavailable:
        selected = [
            record
            for record in records
            if getattr(record.get("gate"), "status", "") in {"PASS", "UNAVAILABLE_FALLBACK"}
        ]
        return selected, rejected, "partial_satisfaction_preserve_upstream_order"

    if passed:
        selected = sorted(passed, key=_record_score, reverse=True)
        return selected, rejected, "quality_gates"

    # All analyzed candidates failed. Keep the strongest few for review instead of
    # crashing or silently emitting nothing.
    ordered = sorted(records, key=_record_score, reverse=True)
    keep_count = max(1, min(len(ordered), int(V3_SATISFACTION_FALLBACK_TOP_K)))
    selected = ordered[:keep_count]
    rejected = ordered[keep_count:]
    for record in selected:
        gate = record.get("gate")
        if gate is not None:
            gate.status = "NEEDS_REVIEW"
            gate.passed = True
            gate.needs_review = True
            if "all_candidates_failed_quality_gates" not in gate.reasons:
                gate.reasons.append("all_candidates_failed_quality_gates")
    return selected, rejected, "all_failed_keep_best_for_review"
