from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ReactionEvent:
    kind: str
    text: str
    start: float
    end: float
    score: float
    source: str = "audio"
    raw_label: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReactionEvent":
        return cls(
            kind=str(payload.get("kind", "reaction")),
            text=str(payload.get("text", "[REACTION]")),
            start=float(payload.get("start", 0.0) or 0.0),
            end=float(payload.get("end", 0.0) or 0.0),
            score=float(payload.get("score", 0.0) or 0.0),
            source=str(payload.get("source", "audio")),
            raw_label=str(payload.get("raw_label", "")),
        )
