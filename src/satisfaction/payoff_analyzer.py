from __future__ import annotations

from src.satisfaction.models import PayoffEvidence
from src.v3.models import EditPlan


def analyze_payoff(raw: dict | None, plan: EditPlan) -> PayoffEvidence:
    raw = raw if isinstance(raw, dict) else {}
    payoff_raw = raw.get("payoff") if isinstance(raw.get("payoff"), dict) else {}

    timestamp = payoff_raw.get("timestamp")
    if timestamp is None:
        cursor = 0.0
        for segment in plan.timeline:
            if segment.purpose in {"payoff", "reaction", "aftermath"}:
                timestamp = cursor
                break
            cursor += segment.duration

    exists = bool(payoff_raw.get("exists", False))
    if not exists:
        exists = any(item.purpose in {"payoff", "reaction", "aftermath"} for item in plan.timeline)

    evidence = PayoffEvidence(
        exists=exists,
        type=str(payoff_raw.get("type", "unknown") or "unknown"),
        timestamp=timestamp,
        strength=payoff_raw.get("strength", raw.get("payoff_score")),
        reason=str(
            payoff_raw.get("reason")
            or (raw.get("score_reasons") or {}).get("payoff_reason")
            or ""
        ),
    )
    return evidence.normalize()
