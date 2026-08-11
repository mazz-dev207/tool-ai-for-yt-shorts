from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ALLOWED_PURPOSES = {
    "cold_open", "hook", "context", "setup", "escalation",
    "payoff", "reaction", "aftermath", "callback", "replay",
}
ALLOWED_HOOK_MODES = {"NATIVE", "RECONSTRUCTED", "EDITORIAL"}
ALLOWED_VISUAL_EFFECTS = {
    "punch_in", "face_zoom", "focus_crop", "freeze_frame",
    "caption_emphasis", "replay", "hard_cut", "none",
}
ALLOWED_AUDIO_EFFECTS = {"impact", "accent", "whoosh", "duck", "none"}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


@dataclass
class HookPlan:
    mode: str = "NATIVE"
    score: int = 0
    text: str = ""
    source_start: float | None = None
    source_end: float | None = None
    duration: float = 0.0
    reason: str = ""
    confidence: float = 0.0

    def validate(self) -> list[str]:
        problems: list[str] = []
        self.mode = str(self.mode or "NATIVE").upper()
        if self.mode not in ALLOWED_HOOK_MODES:
            problems.append("invalid_hook_mode")
        self.score = max(0, min(100, int(round(self.score or 0))))
        self.confidence = _clamp(self.confidence, 0.0, 1.0)
        self.duration = max(0.0, float(self.duration or 0.0))
        if self.source_start is not None and self.source_end is not None:
            if float(self.source_end) <= float(self.source_start):
                problems.append("invalid_hook_source_range")
        return problems


@dataclass
class TimelineSegment:
    source_start: float
    source_end: float
    purpose: str
    preserve_audio: bool = True
    playback_rate: float = 1.0
    semantic_note: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.source_end - self.source_start)

    def validate(self, source_duration: float | None = None) -> list[str]:
        problems: list[str] = []
        self.source_start = float(self.source_start)
        self.source_end = float(self.source_end)
        self.purpose = str(self.purpose or "context").lower()
        self.playback_rate = _clamp(self.playback_rate or 1.0, 0.5, 2.0)
        if abs(self.playback_rate - 1.0) > 0.001:
            problems.append("playback_rate_not_supported_yet")
        if self.source_start < 0 or self.source_end <= self.source_start:
            problems.append("invalid_segment_duration")
        if source_duration is not None and self.source_end > float(source_duration) + 0.001:
            problems.append("segment_outside_source")
        if self.purpose not in ALLOWED_PURPOSES:
            problems.append("invalid_segment_purpose")
        return problems


@dataclass
class VisualEvent:
    time: float
    effect: str
    intensity: float = 0.5
    duration: float = 0.6
    target: str = "auto"
    caption_emphasis: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        problems: list[str] = []
        self.time = max(0.0, float(self.time or 0.0))
        self.duration = _clamp(self.duration or 0.6, 0.05, 2.5)
        self.intensity = _clamp(self.intensity or 0.0, 0.0, 1.0)
        self.effect = str(self.effect or "none").lower()
        if self.effect not in ALLOWED_VISUAL_EFFECTS:
            problems.append("invalid_visual_effect")
        return problems


@dataclass
class AudioEvent:
    time: float
    effect: str
    intensity: float = 0.4
    duration: float = 0.35
    gain_db: float = -12.0
    asset: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        self.time = max(0.0, float(self.time or 0.0))
        self.duration = _clamp(self.duration or 0.35, 0.05, 2.0)
        self.intensity = _clamp(self.intensity or 0.0, 0.0, 1.0)
        self.gain_db = _clamp(self.gain_db, -30.0, -6.0)
        self.effect = str(self.effect or "none").lower()
        if self.effect not in ALLOWED_AUDIO_EFFECTS:
            problems.append("invalid_audio_effect")
        return problems


@dataclass
class ContextOverlay:
    text: str
    start: float
    duration: float
    purpose: str = "context"

    def validate(self) -> list[str]:
        problems: list[str] = []
        self.text = " ".join(str(self.text or "").split()).strip()
        self.start = max(0.0, float(self.start or 0.0))
        self.duration = _clamp(self.duration or 1.2, 0.4, 3.0)
        if not self.text:
            problems.append("empty_overlay")
        if len(self.text) > 120:
            problems.append("overlay_too_long")
        return problems


@dataclass
class CropInstruction:
    target: str = "auto"
    mode: str = "preserve_existing_smartcrop"
    start: float = 0.0
    duration: float = 0.0


@dataclass
class TransitionInstruction:
    kind: str = "hard_cut"
    at: float = 0.0
    duration: float = 0.0


@dataclass
class OriginalityResult:
    originality_score: int = 0
    editorial_transformation: int = 0
    narrative_restructuring: int = 0
    context_independence: int = 0
    visual_transformation: int = 0
    hook_originality: int = 0
    audio_transformation: int = 0
    brand_consistency: int = 0
    source_dependency: int = 0
    recommended_transformations: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def normalize(self) -> "OriginalityResult":
        for name in (
            "originality_score", "editorial_transformation", "narrative_restructuring",
            "context_independence", "visual_transformation", "hook_originality",
            "audio_transformation", "brand_consistency", "source_dependency",
        ):
            setattr(self, name, max(0, min(100, int(round(getattr(self, name, 0) or 0)))))
        return self


@dataclass
class OriginalityQAReport:
    passed: bool
    originality_score: int
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class EditPlan:
    clip_index: int
    hook: HookPlan
    timeline: list[TimelineSegment]
    visual_events: list[VisualEvent] = field(default_factory=list)
    audio_events: list[AudioEvent] = field(default_factory=list)
    context_overlays: list[ContextOverlay] = field(default_factory=list)
    crop_instructions: list[CropInstruction] = field(default_factory=list)
    transitions: list[TransitionInstruction] = field(default_factory=list)
    caption_emphasis: list[dict[str, str]] = field(default_factory=list)
    editorial_angle: str = ""
    viewer_question: str = ""
    stakes: str = ""
    payoff: str = ""
    no_transformation_needed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return round(sum(item.duration for item in self.timeline), 3)

    def validate(
        self,
        source_duration: float | None = None,
        max_segments: int = 8,
        max_effects_per_event: int = 2,
    ) -> list[str]:
        problems = self.hook.validate()
        if not self.timeline:
            problems.append("empty_timeline")
        if len(self.timeline) > max_segments:
            problems.append("excessive_fragmentation")

        seen_ranges: dict[tuple[int, int], str] = {}
        for item in self.timeline:
            problems.extend(item.validate(source_duration))
            key = (int(round(item.source_start * 1000)), int(round(item.source_end * 1000)))
            if key in seen_ranges and item.purpose not in {"cold_open", "replay", "callback"}:
                problems.append("duplicate_content")
            seen_ranges[key] = item.purpose

        for item in self.visual_events:
            problems.extend(item.validate())
        for item in self.audio_events:
            problems.extend(item.validate())
        for item in self.context_overlays:
            problems.extend(item.validate())

        buckets: dict[int, int] = {}
        for event in self.visual_events:
            bucket = int(round(event.time * 2))
            buckets[bucket] = buckets.get(bucket, 0) + 1
        if any(count > max_effects_per_event for count in buckets.values()):
            problems.append("effect_budget_exceeded")

        return sorted(set(problems))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "EditPlan":
        return cls(
            clip_index=int(payload.get("clip_index", 1)),
            hook=HookPlan(**(payload.get("hook") or {})),
            timeline=[TimelineSegment(**item) for item in payload.get("timeline", [])],
            visual_events=[VisualEvent(**item) for item in payload.get("visual_events", [])],
            audio_events=[AudioEvent(**item) for item in payload.get("audio_events", [])],
            context_overlays=[ContextOverlay(**item) for item in payload.get("context_overlays", [])],
            crop_instructions=[CropInstruction(**item) for item in payload.get("crop_instructions", [])],
            transitions=[TransitionInstruction(**item) for item in payload.get("transitions", [])],
            caption_emphasis=list(payload.get("caption_emphasis", []) or []),
            editorial_angle=str(payload.get("editorial_angle", "")),
            viewer_question=str(payload.get("viewer_question", "")),
            stakes=str(payload.get("stakes", "")),
            payoff=str(payload.get("payoff", "")),
            no_transformation_needed=bool(payload.get("no_transformation_needed", False)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )
