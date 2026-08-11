from __future__ import annotations

import json
from pathlib import Path

from src.config import BASE_DIR
from src.strategy.models import ChannelStrategy, FormatBrief, FormatProfile


DEFAULT_GAMING_LIBRARY = BASE_DIR / "config" / "formats" / "mazclips_gaming_formats.json"
DEFAULT_GAMING_STRATEGY = BASE_DIR / "config" / "channel_strategy_mazclips_gaming.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_format_library(path: Path | None = None) -> list[FormatProfile]:
    source = path or DEFAULT_GAMING_LIBRARY
    raw = _load_json(source)
    formats = raw.get("formats", [])
    if not isinstance(formats, list):
        raise ValueError("format library must contain a formats list")
    result: list[FormatProfile] = []
    seen: set[str] = set()
    for item in formats:
        profile = FormatProfile.from_dict(item)
        if not profile.enabled or profile.id in seen:
            continue
        seen.add(profile.id)
        result.append(profile)
    if not result:
        raise ValueError("format library contains no enabled formats")
    return result


def load_channel_strategy(path: Path | None = None) -> ChannelStrategy:
    return ChannelStrategy.from_dict(_load_json(path or DEFAULT_GAMING_STRATEGY))


def load_format_brief(path: Path | None) -> FormatBrief | None:
    if path is None:
        return None
    return FormatBrief.from_dict(_load_json(path))


def resolve_strategy_path(content_profile: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if str(content_profile or "").strip().lower() == "gaming":
        return DEFAULT_GAMING_STRATEGY
    return None


def build_fallback_strategy(content_profile: str) -> ChannelStrategy:
    profile = str(content_profile or "general").strip().lower()
    return ChannelStrategy(
        channel_name=f"MazClips {profile.title()}",
        primary_niche=profile,
        positioning="Self-contained short-form moments with a clear hook and payoff.",
        priority_subniches=[],
        priority_formats=[],
        avoid=["context-heavy clips", "moments without a payoff"],
        min_format_fit=35.0,
        min_content_opportunity=45.0,
    )
