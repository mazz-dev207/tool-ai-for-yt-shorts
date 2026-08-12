from __future__ import annotations


def analyze_expectation_match(raw: dict | None) -> tuple[int | None, str, str, int]:
    raw = raw if isinstance(raw, dict) else {}
    try:
        score = max(0, min(100, int(round(float(raw.get("expectation_match_score"))))))
    except (TypeError, ValueError):
        score = None

    hook_promise = " ".join(str(raw.get("hook_promise", "") or "").split())[:500]
    actual_payoff = " ".join(str(raw.get("actual_payoff", "") or "").split())[:500]
    risks = raw.get("risks") if isinstance(raw.get("risks"), dict) else {}

    explicit_gap = risks.get("clickbait_gap_score")
    try:
        clickbait_gap = max(0, min(100, int(round(float(explicit_gap)))))
    except (TypeError, ValueError):
        clickbait_gap = max(0, 100 - score) if score is not None and bool(risks.get("clickbait_gap", False)) else 0

    return score, hook_promise, actual_payoff, clickbait_gap
