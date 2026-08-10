from __future__ import annotations

from src.config import (
    HOOK_MAX_SHIFT_AFTER,
    HOOK_MAX_SHIFT_BEFORE,
    HOOK_MICRO_PREROLL_MAX,
    HOOK_MICRO_PREROLL_MIN,
    HOOK_SCORE_TIE_THRESHOLD,
)
from src.hooks.models import HookStartCandidate
from src.smart_cut import find_boundary_candidates, flatten_words, snap_to_word_boundary


HOOK_COMPONENT_LIMITS = {
    "immediate_action": 20,
    "curiosity_gap": 20,
    "emotional_reaction": 15,
    "conflict_tension": 15,
    "visual_surprise": 10,
    "context_independence": 10,
    "payoff_proximity": 10,
}

HOOK_PENALTY_LIMITS = {
    "dead_air": (-20, 0),
    "context_dependency": (-20, 0),
    "spoiled_payoff": (-30, 0),
    "mid_sentence": (-15, 0),
    "duplicate_information": (-20, 0),
}

HOOK_TYPES = {
    "ACTION", "REACTION", "CURIOSITY", "CONFLICT", "SURPRISE", "STAKES",
    "QUESTION", "PREDICTION", "PAYOFF_TEASE", "VISUAL_WTF", "MIXED",
}

PROFILE_PRIORITIES = {
    "gaming": "Prioritize action, stakes, conflict, visual surprise and authentic creator reaction.",
    "entertainment": "Prioritize reaction, curiosity, surprise, conflict and a self-contained micro-story.",
    "reaction": "Prioritize emotional reaction, curiosity, visual surprise and context independence.",
    "podcast": "Prioritize strong statements/questions, conflict, curiosity and context independence.",
    "general": "Prioritize minimum context, immediate curiosity, tension and a nearby payoff.",
    "auto": "Infer the content type, then prioritize minimum context, curiosity, tension and payoff.",
}


def _clamp_int(value, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = 0
    return max(minimum, min(maximum, parsed))


def _clamp_float(value, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(minimum, min(maximum, parsed))


def normalize_hook_candidate(raw: dict) -> HookStartCandidate:
    scores = {
        key: _clamp_int((raw.get("hook_scores") or raw.get("scores") or {}).get(key, 0), 0, limit)
        for key, limit in HOOK_COMPONENT_LIMITS.items()
    }
    raw_penalties = raw.get("hook_penalties") or raw.get("penalties") or {}
    penalties = {
        key: _clamp_int(raw_penalties.get(key, 0), bounds[0], bounds[1])
        for key, bounds in HOOK_PENALTY_LIMITS.items()
    }
    component_total = sum(scores.values())
    effective = max(0, min(100, component_total + sum(penalties.values())))
    hook_type = str(raw.get("hook_type", "MIXED") or "MIXED").strip().upper()
    if hook_type not in HOOK_TYPES:
        hook_type = "MIXED"
    secondary = str(raw.get("secondary_hook_type") or "").strip().upper() or None
    if secondary not in HOOK_TYPES:
        secondary = None
    return HookStartCandidate(
        start=round(float(raw.get("start", 0.0)), 3),
        hook_score=effective,
        scores=scores,
        penalties=penalties,
        hook_type=hook_type,
        secondary_hook_type=secondary,
        reason=str(raw.get("hook_reason") or raw.get("reason") or "").strip()[:240],
        confidence=_clamp_float(raw.get("confidence", 0.0), 0.0, 1.0),
        loop_score=_clamp_int(raw.get("loop_score", 0), 0, 100),
        effective_score=effective,
    )


def _segment_start_score(start: float, transcript: list[dict]) -> float:
    score = 0.0
    for segment in transcript or []:
        try:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
        except (TypeError, ValueError):
            continue
        if abs(start - seg_start) <= 0.08:
            score += 2.0
        if seg_start < start < seg_end:
            text = str(segment.get("text", "")).strip().lower()
            if text.startswith(("and ", "but ", "because ", "so then ", "like i was", "as i said")):
                score -= 1.5
    return score


def continuity_score(start: float, transcript: list[dict], original_start: float) -> float:
    words = flatten_words(transcript)
    score = _segment_start_score(start, transcript)
    inside_word = False
    for word in words:
        if word.start + 0.015 < start < word.end - 0.015:
            inside_word = True
            break
        if abs(start - word.start) <= 0.06:
            score += 1.2
            break
    if inside_word:
        score -= 4.0
    score -= min(1.0, abs(start - original_start) * 0.12)
    return round(score, 4)


def select_best_hook_candidate(
    candidates: list[HookStartCandidate],
    transcript: list[dict],
    original_start: float,
) -> HookStartCandidate | None:
    if not candidates:
        return None
    for candidate in candidates:
        candidate.continuity_score = continuity_score(candidate.start, transcript, original_start)
    best_score = max(candidate.hook_score for candidate in candidates)
    close = [
        candidate for candidate in candidates
        if best_score - candidate.hook_score <= HOOK_SCORE_TIE_THRESHOLD
    ]
    return max(
        close,
        key=lambda candidate: (
            candidate.continuity_score,
            candidate.confidence,
            candidate.hook_score,
            -abs(candidate.start - original_start),
        ),
    )


def clamp_semantic_start(start: float, original_start: float, highlight_end: float) -> float:
    lower = max(0.0, original_start - HOOK_MAX_SHIFT_BEFORE)
    upper = min(highlight_end - 0.35, original_start + HOOK_MAX_SHIFT_AFTER)
    return round(max(lower, min(float(start), upper)), 3)


def refine_start_locally(
    semantic_start: float,
    transcript: list[dict],
    original_start: float,
    highlight_end: float,
) -> float:
    """Snap Gemini's semantic zone to a natural audio/text boundary with tiny pre-roll."""
    semantic_start = clamp_semantic_start(semantic_start, original_start, highlight_end)
    words = flatten_words(transcript)
    if not words:
        return semantic_start

    semantic_start = snap_to_word_boundary(semantic_start, "start", words)
    boundaries = find_boundary_candidates(
        semantic_start,
        "start",
        transcript,
        words,
        search_window=max(0.35, HOOK_MICRO_PREROLL_MAX + 0.10),
    )
    viable = [
        item for item in boundaries
        if semantic_start - HOOK_MICRO_PREROLL_MAX <= item.time <= semantic_start + 0.12
    ]
    if viable:
        best = max(
            viable,
            key=lambda item: item.score - abs(item.time - semantic_start) * 0.45,
        )
        adjusted = best.time
    else:
        adjusted = semantic_start

    # Add only a tiny natural pre-roll when it does not enter another spoken word.
    preroll = max(0.0, min(HOOK_MICRO_PREROLL_MAX, HOOK_MICRO_PREROLL_MIN))
    if preroll > 0:
        proposed = max(0.0, adjusted - preroll)
        cuts_previous_word = any(
            word.start + 0.015 < proposed < word.end - 0.015
            for word in words
        )
        if not cuts_previous_word:
            adjusted = proposed

    return clamp_semantic_start(adjusted, original_start, highlight_end)


def profile_priority_text(profile: str | None) -> str:
    return PROFILE_PRIORITIES.get(str(profile or "auto").strip().lower(), PROFILE_PRIORITIES["general"])
