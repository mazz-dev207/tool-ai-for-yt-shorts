from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _score(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


@dataclass
class PayoffEvidence:
    exists: bool = False
    type: str = "unknown"
    timestamp: float | None = None
    strength: int | None = None
    reason: str = ""

    def normalize(self) -> "PayoffEvidence":
        self.type = str(self.type or "unknown").strip().lower()[:80]
        if self.timestamp is not None:
            try:
                self.timestamp = max(0.0, float(self.timestamp))
            except (TypeError, ValueError):
                self.timestamp = None
        self.strength = _score(self.strength)
        self.reason = " ".join(str(self.reason or "").split())[:600]
        return self


@dataclass
class SatisfactionStructure:
    setup: bool = False
    tension: bool = False
    escalation: bool = False
    payoff: bool = False
    pattern: str = "unknown"


@dataclass
class SatisfactionRisks:
    confusing_start: bool = False
    missing_context: bool = False
    weak_payoff: bool = False
    clickbait_gap: bool = False
    abrupt_ending: bool = False
    dead_air: bool = False
    generic_outro: bool = False
    clickbait_gap_score: int = 0

    def normalize(self) -> "SatisfactionRisks":
        self.clickbait_gap_score = _score(self.clickbait_gap_score) or 0
        return self


@dataclass
class ProtectedRange:
    start: float
    end: float
    reason: str

    def normalize(self) -> "ProtectedRange":
        self.start = max(0.0, float(self.start))
        self.end = max(self.start, float(self.end))
        self.reason = " ".join(str(self.reason or "").split())[:160]
        return self


@dataclass
class EndingCandidate:
    end: float
    payoff_score: int | None = None
    emotional_completeness_score: int | None = None
    ending_quality_score: int | None = None
    viewer_satisfaction_score: int | None = None
    reason: str = ""

    def normalize(self) -> "EndingCandidate":
        self.end = max(0.0, float(self.end))
        self.payoff_score = _score(self.payoff_score)
        self.emotional_completeness_score = _score(self.emotional_completeness_score)
        self.ending_quality_score = _score(self.ending_quality_score)
        self.viewer_satisfaction_score = _score(self.viewer_satisfaction_score)
        self.reason = " ".join(str(self.reason or "").split())[:400]
        return self


@dataclass
class SatisfactionResult:
    status: str = "unavailable"
    viewer_satisfaction_score: int | None = None
    payoff_score: int | None = None
    expectation_match_score: int | None = None
    context_independence_score: int | None = None
    clarity_score: int | None = None
    emotional_completeness_score: int | None = None
    value_density_score: int | None = None
    ending_quality_score: int | None = None
    structure: SatisfactionStructure = field(default_factory=SatisfactionStructure)
    payoff: PayoffEvidence = field(default_factory=PayoffEvidence)
    risks: SatisfactionRisks = field(default_factory=SatisfactionRisks)
    reasons: dict[str, str] = field(default_factory=dict)
    hook_promise: str = ""
    actual_payoff: str = ""
    recommended_changes: list[str] = field(default_factory=list)
    protected_ranges: list[ProtectedRange] = field(default_factory=list)
    ending_candidates: list[EndingCandidate] = field(default_factory=list)
    recommended_end: float | None = None
    end_adjustment_applied: bool = False
    end_adjustment_reason: str = ""
    evidence_source: str = "none"

    @property
    def available(self) -> bool:
        return self.status in {"available", "partial"} and self.viewer_satisfaction_score is not None

    def normalize(self) -> "SatisfactionResult":
        for name in (
            "viewer_satisfaction_score",
            "payoff_score",
            "expectation_match_score",
            "context_independence_score",
            "clarity_score",
            "emotional_completeness_score",
            "value_density_score",
            "ending_quality_score",
        ):
            setattr(self, name, _score(getattr(self, name)))
        self.status = str(self.status or "unavailable").strip().lower()
        self.hook_promise = " ".join(str(self.hook_promise or "").split())[:500]
        self.actual_payoff = " ".join(str(self.actual_payoff or "").split())[:500]
        self.reasons = {
            str(key)[:80]: " ".join(str(value or "").split())[:600]
            for key, value in (self.reasons or {}).items()
            if str(value or "").strip()
        }
        self.recommended_changes = [
            " ".join(str(item).split())[:240]
            for item in (self.recommended_changes or [])
            if str(item).strip()
        ][:8]
        self.protected_ranges = [item.normalize() for item in self.protected_ranges if item.end > item.start]
        self.ending_candidates = [item.normalize() for item in self.ending_candidates]
        if self.recommended_end is not None:
            try:
                self.recommended_end = max(0.0, float(self.recommended_end))
            except (TypeError, ValueError):
                self.recommended_end = None
        self.risks.normalize()
        self.payoff.normalize()
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityGateResult:
    status: str = "UNAVAILABLE_FALLBACK"
    passed: bool = True
    reasons: list[str] = field(default_factory=list)
    needs_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinalScoreResult:
    score: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
    unavailable_components: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
