from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import HIGHLIGHTS_DIR, TRANSCRIPT_DIR
from src.logger import info, success, warning
from src.strategy.library import (
    build_fallback_strategy,
    load_channel_strategy,
    load_format_brief,
    load_format_library,
    resolve_strategy_path,
)
from src.strategy.models import (
    ChannelStrategy,
    ContentOpportunityScore,
    FormatBrief,
    FormatMatch,
    FormatProfile,
)


_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "clear_goal": (
        "trying to", "try to", "need to", "have to", "gotta", "goal", "challenge",
        "attempt", "mission", "objective", "win", "beat", "finish", "survive",
    ),
    "stakes_or_risk": (
        "last chance", "one attempt", "only attempt", "1 hp", "one hp", "low hp",
        "risk", "lose", "lost", "bet", "money", "dollar", "time left", "last life",
        "final round", "match point", "rank", "eliminated",
    ),
    "failure_or_reversal": (
        "fail", "failed", "failure", "miss", "missed", "wrong", "mistake", "died",
        "dead", "lost", "no way", "what", "oh no", "threw", "choked", "backfired",
    ),
    "reaction": (
        "no way", "what", "wtf", "wow", "oh my", "omg", "bro", "dude", "laugh",
        "haha", "scream", "rage", "mad", "angry", "crazy", "insane",
    ),
    "bad_position": (
        "low hp", "1 hp", "one hp", "outnumbered", "behind", "losing", "trapped",
        "no ammo", "reload", "last player", "last alive", "almost dead",
    ),
    "escalation": (
        "then", "again", "another", "pushing", "coming", "closer", "last", "final",
        "more", "worse", "suddenly", "now",
    ),
    "clutch_or_recovery": (
        "clutch", "comeback", "recovered", "somehow won", "won", "win", "saved",
        "survived", "last second", "turned it around",
    ),
    "payoff": (
        "won", "lost", "died", "killed", "clutch", "failed", "worked", "did it",
        "got it", "finally", "result", "payoff",
    ),
    "normal_state": (
        "normal", "just", "walking", "playing", "waiting", "nothing", "quiet",
    ),
    "visual_surprise": (
        "suddenly", "appeared", "spawned", "glitch", "bug", "what is that", "what was that",
        "unexpected", "came out of nowhere", "behind me", "behind him",
    ),
    "reaction_or_consequence": (
        "reaction", "no way", "what", "died", "lost", "killed", "laughed", "screamed",
    ),
    "social_interaction": (
        "he said", "she said", "they said", "told me", "asked", "bro", "dude", "teammate",
        "player", "friend", "voice chat", "chat", "talking",
    ),
    "misunderstanding_or_conflict": (
        "argue", "argument", "conflict", "thought", "misunderstood", "why", "stop", "no",
        "you did", "he did", "she did", "mad",
    ),
    "punchline_or_reaction": (
        "laugh", "haha", "funny", "joke", "no way", "what", "bro", "dude",
    ),
    "constraint": (
        "one attempt", "only", "without", "before", "under", "limit", "limited", "time",
        "seconds", "minutes", "last chance", "one life",
    ),
    "success_or_failure": (
        "success", "won", "win", "did it", "worked", "failed", "lost", "died", "missed",
    ),
    "expectation": (
        "should", "supposed", "expected", "thought", "easy", "sure", "definitely",
    ),
    "negative_reversal": (
        "but", "instead", "suddenly", "failed", "died", "lost", "wrong", "backfired",
        "choked", "threw",
    ),
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _score(candidate: dict, key: str, default: float = 50.0) -> float:
    raw = candidate.get("scores") or {}
    try:
        return max(0.0, min(100.0, float(raw.get(key, default) or default)))
    except (TypeError, ValueError):
        return default


def _candidate_text(candidate: dict, transcript: list[dict]) -> str:
    parts = [
        str(candidate.get("title", "")),
        str(candidate.get("hook", "")),
        str(candidate.get("payoff", "")),
        str(candidate.get("reason", "")),
    ]
    try:
        start = float(candidate["start"])
        end = float(candidate["end"])
    except (KeyError, TypeError, ValueError):
        start, end = 0.0, 0.0
    for segment in transcript or []:
        try:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
        except (TypeError, ValueError):
            continue
        if seg_end < start - 1.0 or seg_start > end + 1.0:
            continue
        parts.append(str(segment.get("text", "")))
    return " ".join(parts).lower()


def detect_signals(candidate: dict, transcript: list[dict]) -> set[str]:
    text = _candidate_text(candidate, transcript)
    text = re.sub(r"\s+", " ", text)
    signals: set[str] = set()
    for signal, patterns in _SIGNAL_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            signals.add(signal)

    if _score(candidate, "emotional_intensity", 0) >= 65:
        signals.add("reaction")
        signals.add("punchline_or_reaction")
    if _score(candidate, "payoff", 0) >= 65:
        signals.add("payoff")
        signals.add("success_or_failure")
        signals.add("reaction_or_consequence")
    if _score(candidate, "entertainment", 0) >= 72:
        signals.add("visual_surprise")
    if _score(candidate, "retention", 0) >= 72 and _score(candidate, "pacing", 0) >= 65:
        signals.add("escalation")
    if str(candidate.get("hook", "")).strip():
        signals.add("clear_goal" if any(word in text for word in ("try", "need", "goal", "attempt", "challenge")) else "normal_state")
    return signals


def _duration_fit(duration: float, profile: FormatProfile) -> float:
    low, high = profile.target_duration_seconds
    if low <= duration <= high:
        return 100.0
    distance = low - duration if duration < low else duration - high
    return max(0.0, 100.0 - distance * 8.0)


def score_format_match(
    candidate: dict,
    transcript: list[dict],
    profile: FormatProfile,
    strategy: ChannelStrategy,
    brief: FormatBrief | None,
) -> FormatMatch:
    signals = detect_signals(candidate, transcript)
    required = list(profile.required_signals)
    present = [item for item in required if item in signals]
    missing = [item for item in required if item not in signals]
    coverage = 100.0 if not required else (len(present) / len(required)) * 100.0

    semantic_strength = (
        _score(candidate, "hook") * 0.20
        + _score(candidate, "payoff") * 0.25
        + _score(candidate, "standalone_context") * 0.20
        + _score(candidate, "retention") * 0.20
        + _score(candidate, "emotional_intensity") * 0.15
    )
    duration = max(0.0, float(candidate.get("end", 0)) - float(candidate.get("start", 0)))
    duration_fit = _duration_fit(duration, profile)
    fit = coverage * 0.58 + semantic_strength * 0.32 + duration_fit * 0.10

    priorities = list(strategy.priority_formats)
    if brief:
        priorities = list(dict.fromkeys([*brief.priority_formats, *priorities]))
    if profile.id in priorities:
        fit += 4.0
    if profile.niche != strategy.primary_niche and strategy.primary_niche not in {"", "auto", "general"}:
        fit -= 15.0
    fit = max(0.0, min(100.0, fit))

    if fit >= strategy.min_format_fit:
        recommendation = "use"
    elif fit >= max(25.0, strategy.min_format_fit - 15.0):
        recommendation = "low_priority"
    else:
        recommendation = "skip"

    reason = (
        f"{len(present)}/{len(required)} required signals present; "
        f"semantic_strength={semantic_strength:.0f}, duration_fit={duration_fit:.0f}."
    )
    return FormatMatch(
        format_id=profile.id,
        format_version=profile.version,
        format_fit_score=fit,
        present_signals=present,
        missing_signals=missing,
        recommendation=recommendation,
        reason=reason,
    )


def choose_best_format(
    candidate: dict,
    transcript: list[dict],
    formats: list[FormatProfile],
    strategy: ChannelStrategy,
    brief: FormatBrief | None,
    manual_format_id: str | None = None,
) -> tuple[FormatProfile, FormatMatch, list[FormatMatch]]:
    available = [item for item in formats if item.enabled]
    if manual_format_id:
        available = [item for item in available if item.id == manual_format_id]
        if not available:
            raise ValueError(f"Unknown or disabled format id: {manual_format_id}")
    matches = [
        score_format_match(candidate, transcript, item, strategy, brief)
        for item in available
    ]
    if not matches:
        raise ValueError("No formats available for matching")
    best_match = max(matches, key=lambda item: item.format_fit_score)
    profile = next(item for item in available if item.id == best_match.format_id)
    return profile, best_match, sorted(matches, key=lambda item: item.format_fit_score, reverse=True)


def calculate_content_opportunity(
    candidate: dict,
    profile: FormatProfile,
    match: FormatMatch,
    strategy: ChannelStrategy,
    brief: FormatBrief | None,
) -> ContentOpportunityScore:
    channel_fit = 100.0 if profile.niche == strategy.primary_niche else 45.0
    if profile.id in strategy.priority_formats:
        channel_fit = 100.0
    transformability = (
        _score(candidate, "standalone_context") * 0.35
        + _score(candidate, "payoff") * 0.35
        + _score(candidate, "hook") * 0.30
    )
    emotional_visual = (
        _score(candidate, "emotional_intensity") * 0.55
        + _score(candidate, "entertainment") * 0.45
    )
    components = {
        "format_match": match.format_fit_score,
        "hook_potential": _score(candidate, "hook"),
        "payoff_strength": _score(candidate, "payoff"),
        "context_independence": _score(candidate, "standalone_context"),
        "emotional_visual_signal": emotional_visual,
        "source_quality": _score(candidate, "viral_potential"),
        "transformability": transformability,
        "channel_fit": channel_fit,
    }
    total = (
        components["format_match"] * 0.25
        + components["hook_potential"] * 0.15
        + components["payoff_strength"] * 0.15
        + components["context_independence"] * 0.10
        + components["emotional_visual_signal"] * 0.10
        + components["source_quality"] * 0.10
        + components["transformability"] * 0.10
        + components["channel_fit"] * 0.05
    )
    demand = brief.external_demand_signal if brief else None
    return ContentOpportunityScore(total=total, components=components, external_demand_signal=demand)


def apply_format_intelligence(
    video_name: str,
    content_profile: str,
    *,
    strategy_path: str | None = None,
    format_brief_path: str | None = None,
    manual_format_id: str | None = None,
    skip_low_fit: bool = False,
) -> Path:
    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"
    if not highlights_path.exists():
        raise FileNotFoundError(highlights_path)
    candidates = _load_json(highlights_path)
    transcript = _load_json(transcript_path) if transcript_path.exists() else []
    if not isinstance(candidates, list):
        raise ValueError("highlights JSON must contain a list")

    resolved_strategy = resolve_strategy_path(content_profile, strategy_path)
    if resolved_strategy is not None:
        try:
            strategy = load_channel_strategy(resolved_strategy)
        except Exception as exc:
            warning(f"[FORMAT] Strategy load failed: {exc}; using fallback strategy.")
            strategy = build_fallback_strategy(content_profile)
    else:
        strategy = build_fallback_strategy(content_profile)

    brief = None
    if format_brief_path:
        try:
            brief = load_format_brief(Path(format_brief_path).expanduser().resolve())
        except Exception as exc:
            warning(f"[FORMAT] FormatBrief invalid: {exc}; continuing without imported demand/reference data.")

    formats = load_format_library()
    if strategy.primary_niche not in {"gaming", "auto", "general"}:
        matching_niche = [item for item in formats if item.niche == strategy.primary_niche]
        if matching_niche:
            formats = matching_niche

    annotated: list[dict] = []
    skipped: list[dict] = []
    metadata: list[dict] = []
    info(
        f"[FORMAT] Strategy={strategy.channel_name} | formats={len(formats)} | "
        f"mode={'manual' if manual_format_id else ('brief' if brief else 'auto')}"
    )

    for index, raw_candidate in enumerate(candidates, start=1):
        candidate = dict(raw_candidate)
        try:
            profile, match, all_matches = choose_best_format(
                candidate,
                transcript if isinstance(transcript, list) else [],
                formats,
                strategy,
                brief,
                manual_format_id=manual_format_id,
            )
            opportunity = calculate_content_opportunity(candidate, profile, match, strategy, brief)
        except Exception as exc:
            warning(f"[FORMAT {index}/{len(candidates)}] analysis failed: {exc}; keeping candidate.")
            annotated.append(candidate)
            continue

        candidate["format_id"] = profile.id
        candidate["format_version"] = profile.version
        candidate["format_fit_score"] = match.format_fit_score
        candidate["content_opportunity_score"] = opportunity.total
        candidate["format_match"] = match.to_dict()
        candidate["strategy_context"] = {
            "channel_strategy": {
                "channel_name": strategy.channel_name,
                "primary_niche": strategy.primary_niche,
                "positioning": strategy.positioning,
            },
            "format_profile": profile.to_dict(),
            "format_match": match.to_dict(),
            "content_opportunity": opportunity.to_dict(),
            "external_demand_claimed": brief.external_demand_signal is not None if brief else False,
        }

        gate_passed = (
            match.format_fit_score >= strategy.min_format_fit
            and opportunity.total >= strategy.min_content_opportunity
        )
        gate = "PASS" if gate_passed else "LOW FORMAT FIT"
        candidate["strategic_gate"] = gate
        info(
            f"[FORMAT {index}/{len(candidates)}] {profile.id} | "
            f"fit={match.format_fit_score:.0f} opportunity={opportunity.total:.0f} | {gate}"
        )

        metadata.append({
            "candidate_index": index,
            "format_profile": profile.to_dict(),
            "format_match": match.to_dict(),
            "all_format_matches": [item.to_dict() for item in all_matches],
            "content_opportunity": opportunity.to_dict(),
            "strategic_gate": gate,
        })
        if skip_low_fit and not gate_passed:
            skipped.append(candidate)
        else:
            annotated.append(candidate)

    annotated.sort(
        key=lambda item: (
            float(item.get("content_opportunity_score", 0) or 0),
            float((item.get("scores") or {}).get("viral_potential", 0) or 0),
        ),
        reverse=True,
    )
    for rank, item in enumerate(annotated, start=1):
        item["strategy_rank"] = rank

    if not annotated and candidates:
        warning("[FORMAT] Gate would remove every candidate; safe fallback keeps original candidates.")
        annotated = candidates
        skipped = []

    _save_json(highlights_path, annotated)
    output = HIGHLIGHTS_DIR / f"{video_name}_format_metadata.json"
    _save_json(output, {
        "channel_strategy": strategy.to_dict(),
        "format_brief": brief.to_dict() if brief else None,
        "manual_format_id": manual_format_id,
        "skip_low_fit": skip_low_fit,
        "input_candidates": len(candidates),
        "output_candidates": len(annotated),
        "skipped_candidates": len(skipped),
        "candidates": metadata,
    })
    success(f"[FORMAT] Format intelligence complete: {len(candidates)} -> {len(annotated)} candidates.")
    return highlights_path
