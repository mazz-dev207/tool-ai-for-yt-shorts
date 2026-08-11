from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EditorialAngle:
    primary_angle: str = "moment"
    secondary_angle: str = ""
    viewer_question: str = ""
    stakes: str = ""
    payoff: str = ""
    reason: str = ""
    confidence: float = 0.0


@dataclass
class OriginalityAnalysis:
    why_interesting: str = ""
    source_dependency: int = 50
    context_independence: int = 50
    current_opening_quality: int = 50
    transformation_need: int = 50
    recommended_transformations: list[str] = field(default_factory=list)
    no_transformation_needed: bool = False
