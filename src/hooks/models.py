from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class HookStartCandidate:
    start: float
    base_hook_score: int
    final_hook_score: int
    core_scores: dict[str, int] = field(default_factory=dict)
    human_modifiers: dict[str, int] = field(default_factory=dict)
    penalties: dict[str, int] = field(default_factory=dict)
    hook_type: str = "MIXED"
    secondary_hook_types: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    loop_score: int = 0
    continuity_score: float = 0.0

    @property
    def hook_score(self) -> int:
        return self.final_hook_score

    @property
    def naturalness(self) -> int:
        return int(self.human_modifiers.get("naturalness", 0) or 0)

    @property
    def forced_hook_penalty(self) -> int:
        keys = (
            "forced_hook",
            "generic_setup",
            "forced_intro",
            "youtuber_intro",
            "fake_hype",
            "obvious_clickbait",
            "repeated_context",
            "fake_urgency",
        )
        return sum(int(self.penalties.get(key, 0) or 0) for key in keys)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HookOptimizationResult:
    original_start: float
    original_end: float
    semantic_start: float
    optimized_start: float
    base_hook_score: int = 0
    hook_score: int = 0
    hook_type: str = "MIXED"
    secondary_hook_types: list[str] = field(default_factory=list)
    confidence: float = 0.0
    hook_reason: str = ""
    core_scores: dict[str, int] = field(default_factory=dict)
    human_modifiers: dict[str, int] = field(default_factory=dict)
    penalties: dict[str, int] = field(default_factory=dict)
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
