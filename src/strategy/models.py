from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


@dataclass(frozen=True)
class FormatProfile:
    id: str
    niche: str
    subniche: str
    format_name: str
    description: str
    required_signals: list[str] = field(default_factory=list)
    preferred_hook_types: list[str] = field(default_factory=list)
    preferred_story_shape: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    target_duration_seconds: tuple[float, float] = (12.0, 30.0)
    version: int = 1
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> "FormatProfile":
        if not isinstance(raw, dict):
            raise ValueError("FormatProfile must be an object")
        format_id = str(raw.get("id", "")).strip()
        if not format_id:
            raise ValueError("FormatProfile.id is required")
        duration = raw.get("target_duration_seconds", [12, 30])
        if not isinstance(duration, (list, tuple)) or len(duration) != 2:
            duration = [12, 30]
        low = max(1.0, float(duration[0]))
        high = max(low, float(duration[1]))
        return cls(
            id=format_id,
            niche=str(raw.get("niche", "general") or "general").strip().lower(),
            subniche=str(raw.get("subniche", "general") or "general").strip().lower(),
            format_name=str(raw.get("format_name", format_id) or format_id).strip(),
            description=str(raw.get("description", "") or "").strip(),
            required_signals=_strings(raw.get("required_signals")),
            preferred_hook_types=_strings(raw.get("preferred_hook_types")),
            preferred_story_shape=_strings(raw.get("preferred_story_shape")),
            avoid=_strings(raw.get("avoid")),
            target_duration_seconds=(low, high),
            version=max(1, int(raw.get("version", 1) or 1)),
            enabled=bool(raw.get("enabled", True)),
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        value["target_duration_seconds"] = list(self.target_duration_seconds)
        return value


@dataclass(frozen=True)
class ChannelStrategy:
    channel_name: str
    primary_niche: str
    positioning: str
    priority_subniches: list[str] = field(default_factory=list)
    priority_formats: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    min_format_fit: float = 45.0
    min_content_opportunity: float = 50.0
    version: int = 1

    @classmethod
    def from_dict(cls, raw: dict) -> "ChannelStrategy":
        if not isinstance(raw, dict):
            raise ValueError("ChannelStrategy must be an object")
        return cls(
            channel_name=str(raw.get("channel_name", "MazClips") or "MazClips").strip(),
            primary_niche=str(raw.get("primary_niche", "general") or "general").strip().lower(),
            positioning=str(raw.get("positioning", "") or "").strip(),
            priority_subniches=_strings(raw.get("priority_subniches")),
            priority_formats=_strings(raw.get("priority_formats")),
            avoid=_strings(raw.get("avoid")),
            min_format_fit=_clamp(float(raw.get("min_format_fit", 45) or 45)),
            min_content_opportunity=_clamp(float(raw.get("min_content_opportunity", 50) or 50)),
            version=max(1, int(raw.get("version", 1) or 1)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FormatBrief:
    channel: str = ""
    target_niche: str = ""
    target_subniche: str = ""
    priority_formats: list[str] = field(default_factory=list)
    reference_patterns: list[dict] = field(default_factory=list)
    source: str = "manual"
    external_demand_signal: float | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "FormatBrief":
        if not isinstance(raw, dict):
            raise ValueError("FormatBrief must be an object")
        references = raw.get("reference_patterns", [])
        if not isinstance(references, list):
            references = []
        demand = raw.get("external_demand_signal")
        try:
            demand_value = None if demand is None else _clamp(float(demand))
        except (TypeError, ValueError):
            demand_value = None
        return cls(
            channel=str(raw.get("channel", "") or "").strip(),
            target_niche=str(raw.get("target_niche", "") or "").strip().lower(),
            target_subniche=str(raw.get("target_subniche", "") or "").strip().lower(),
            priority_formats=_strings(raw.get("priority_formats")),
            reference_patterns=[item for item in references if isinstance(item, dict)],
            source=str(raw.get("source", "manual") or "manual").strip(),
            external_demand_signal=demand_value,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FormatMatch:
    format_id: str
    format_version: int
    format_fit_score: float
    present_signals: list[str]
    missing_signals: list[str]
    recommendation: str
    reason: str

    def __post_init__(self) -> None:
        self.format_fit_score = round(_clamp(self.format_fit_score), 2)
        if self.recommendation not in {"use", "low_priority", "skip"}:
            self.recommendation = "low_priority"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContentOpportunityScore:
    total: float
    components: dict[str, float]
    external_demand_signal: float | None = None

    def __post_init__(self) -> None:
        self.total = round(_clamp(self.total), 2)
        self.components = {
            str(key): round(_clamp(value), 2)
            for key, value in self.components.items()
        }
        if self.external_demand_signal is not None:
            self.external_demand_signal = round(_clamp(self.external_demand_signal), 2)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentMetadata:
    short_id: str
    format_id: str
    format_version: int
    content_opportunity_score: float
    hook_type: str
    hook_score: float
    originality_score: float
    duration: float
    performance: dict[str, float | int | None] = field(default_factory=lambda: {
        "views": None,
        "viewed_vs_swiped": None,
        "average_view_duration": None,
        "average_percentage_viewed": None,
        "likes": None,
        "comments": None,
        "shares": None,
        "subscribers_gained": None,
    })

    def to_dict(self) -> dict:
        return asdict(self)
