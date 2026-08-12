from __future__ import annotations


def analyze_context_independence(
    raw: dict | None,
    *,
    originality_context_score: int | None = None,
) -> tuple[int | None, str]:
    raw = raw if isinstance(raw, dict) else {}
    try:
        score = max(
            0,
            min(100, int(round(float(raw.get("context_independence_score"))))),
        )
    except (TypeError, ValueError):
        score = None

    # Originality already estimates source dependency/context independence. It is
    # useful as supporting evidence, but it never manufactures a full Viewer
    # Satisfaction result when Gemini satisfaction analysis is unavailable.
    if score is None and originality_context_score is not None:
        try:
            score = max(0, min(100, int(round(float(originality_context_score)))))
        except (TypeError, ValueError):
            score = None

    reasons = raw.get("score_reasons") if isinstance(raw.get("score_reasons"), dict) else {}
    reason = " ".join(
        str(
            reasons.get("context_independence_reason")
            or reasons.get("context_reason")
            or ""
        ).split()
    )[:600]
    return score, reason
