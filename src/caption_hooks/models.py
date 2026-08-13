from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CaptionHookCandidate:
    text: str
    type: str = "none"
    mode: str = "COMPLEMENT"
    reason: str = ""
    truth_basis: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    rejected_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaptionHookResult:
    enabled: bool = False
    status: str = "none"
    mode: str = "NONE"
    type: str = "none"
    text: str = ""
    lines: list[str] = field(default_factory=list)
    emphasis: list[str] = field(default_factory=list)
    start: float = 0.05
    duration: float = 1.4
    position: str = "top_safe"
    score: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    source: str = "none"
    cache_hit: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaptionHookResult":
        allowed = {
            "enabled", "status", "mode", "type", "text", "lines", "emphasis",
            "start", "duration", "position", "score", "scores", "penalties",
            "reason", "source", "cache_hit", "candidates",
        }
        return cls(**{key: value for key, value in payload.items() if key in allowed})
