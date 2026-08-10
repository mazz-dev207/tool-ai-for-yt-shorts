from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class HookStartCandidate:
    start: float
    hook_score: int
    scores: dict[str, int] = field(default_factory=dict)
    penalties: dict[str, int] = field(default_factory=dict)
    hook_type: str = "MIXED"
    secondary_hook_type: str | None = None
    reason: str = ""
    confidence: float = 0.0
    loop_score: int = 0
    continuity_score: float = 0.0
    effective_score: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HookOptimizationResult:
    original_start: float
    original_end: float
    semantic_start: float
    optimized_start: float
    hook_score: int = 0
    hook_type: str = "MIXED"
    secondary_hook_type: str | None = None
    confidence: float = 0.0
    hook_reason: str = ""
    hook_scores: dict[str, int] = field(default_factory=dict)
    hook_penalties: dict[str, int] = field(default_factory=dict)
    loop_score: int = 0
    candidates: list[dict] = field(default_factory=list)
    from_cache: bool = False
    applied: bool = False
    fallback_reason: str | None = None

    @property
    def hook_shift(self) -> float:
        return round(self.optimized_start - self.original_start, 3)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["hook_shift"] = self.hook_shift
        return payload
