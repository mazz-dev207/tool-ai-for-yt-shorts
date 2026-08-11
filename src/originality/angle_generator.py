from __future__ import annotations

from src.originality.models import EditorialAngle


def generate_angle(proposal: dict, fallback_reason: str = "") -> EditorialAngle:
    raw = proposal.get("editorial_angle") or {}
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))
    primary = " ".join(str(raw.get("primary_angle") or "moment").split()).strip()
    return EditorialAngle(
        primary_angle=primary or "moment",
        secondary_angle=" ".join(str(raw.get("secondary_angle") or "").split()).strip(),
        viewer_question=" ".join(str(raw.get("viewer_question") or "").split()).strip(),
        stakes=" ".join(str(raw.get("stakes") or "").split()).strip(),
        payoff=" ".join(str(raw.get("payoff") or "").split()).strip(),
        reason=" ".join(str(raw.get("reason") or fallback_reason).split()).strip()[:240],
        confidence=confidence,
    )
