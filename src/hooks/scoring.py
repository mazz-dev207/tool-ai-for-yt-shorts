from __future__ import annotations

from src.config import (
    HOOK_FORCED_INTRO_PENALTY_ENABLED,
    HOOK_MAX_POSITIVE_MODIFIER,
    HOOK_MAX_SHIFT_AFTER,
    HOOK_MAX_SHIFT_BEFORE,
    HOOK_MICRO_PREROLL_MAX,
    HOOK_MICRO_PREROLL_MIN,
    HOOK_NATURALNESS_ENABLED,
    HOOK_SCORE_TIE_THRESHOLD,
)
from src.hooks.models import HookStartCandidate
from src.smart_cut import find_boundary_candidates, flatten_words, snap_to_word_boundary


CORE_HOOK_LIMITS = {
    "immediate_action": 20,
    "curiosity_gap": 20,
    "emotional_reaction": 15,
    "conflict_tension": 15,
    "visual_surprise": 10,
    "context_independence": 10,
    "payoff_proximity": 10,
}

HUMAN_MODIFIER_LIMITS = {
    "contradiction": (-5, 8),
    "specificity": (0, 6),
    "timeframe_tension": (0, 5),
    "relatability": (0, 5),
    "naturalness": (-15, 10),
}

PENALTY_LIMITS = {
    "dead_air": (-20, 0),
    "context_dependency": (-20, 0),
    "spoiled_payoff": (-30, 0),
    "mid_sentence": (-15, 0),
    "duplicate_information": (-15, 0),
    "forced_hook": (-25, 0),
    "generic_setup": (-15, 0),
    "forced_intro": (-20, 0),
    "youtuber_intro": (-25, 0),
    "fake_hype": (-20, 0),
    "obvious_clickbait": (-15, 0),
    "repeated_context": (-15, 0),
    "fake_urgency": (-15, 0),
}

FORCED_PENALTY_KEYS = {
    "forced_hook",
    "generic_setup",
    "forced_intro",
    "youtuber_intro",
    "fake_hype",
    "obvious_clickbait",
    "repeated_context",
    "fake_urgency",
}

HOOK_TYPES = {
    "ACTION",
    "REACTION",
    "CURIOSITY",
    "CONFLICT",
    "SURPRISE",
    "STAKES",
    "QUESTION",
    "PREDICTION",
    "PAYOFF_TEASE",
    "VISUAL_WTF",
    "CONTRADICTION",
    "SPECIFICITY",
    "TIMEFRAME",
    "RELATABLE",
    "NATURAL_REACTION",
    "MIXED",
}

PROFILE_PRIORITIES = {
    "gaming": (
        "Prioritize immediate action, stakes, conflict, visual surprise, authentic reaction, "
        "naturalness and specific game state such as low HP, enemies pushing, clutch/fail or panic reload."
    ),
    "entertainment": (
        "Prioritize curiosity, contradiction, specificity, surprise, relatability, naturalness and a complete micro-story."
    ),
    "reaction": (
        "Prioritize emotional reaction, curiosity, naturalness, visual surprise, specificity, relatability and context independence."
    ),
    "podcast": (
        "Prioritize unexpected statements, contradiction, specificity, conflict, questions, curiosity, naturalness and context independence."
    ),
    "general": (
        "Prioritize minimum context, maximum curiosity, clear stakes, naturalness and a nearby unresolved payoff."
    ),
    "auto": (
        "Infer the content type, then prioritize minimum context, maximum curiosity, clear stakes, naturalness and payoff."
    ),
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


def _normalize_penalty(value, minimum: int) -> int:
    """Store penalties as negative numbers even if a model returns a positive magnitude."""
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    if parsed > 0:
        parsed = -parsed
    return max(minimum, min(0, parsed))


def calculate_hook_scores(
    core_scores: dict[str, int],
    human_modifiers: dict[str, int],
    penalties: dict[str, int],
) -> tuple[int, int]:
    """Return (base_hook_score, final_hook_score) with Core Signals dominant."""
    base = sum(int(core_scores.get(key, 0) or 0) for key in CORE_HOOK_LIMITS)
    base = max(0, min(100, base))

    positive_modifiers = sum(max(0, int(value or 0)) for value in human_modifiers.values())
    positive_modifiers = min(max(0, int(HOOK_MAX_POSITIVE_MODIFIER)), positive_modifiers)
    negative_modifiers = sum(min(0, int(value or 0)) for value in human_modifiers.values())
    penalty_total = sum(min(0, int(value or 0)) for value in penalties.values())

    final = base + positive_modifiers + negative_modifiers + penalty_total
    return base, max(0, min(100, int(round(final))))


def normalize_hook_candidate(raw: dict) -> HookStartCandidate:
    raw_core = raw.get("core_scores") or raw.get("hook_scores") or raw.get("scores") or {}
    core_scores = {
        key: _clamp_int(raw_core.get(key, 0), 0, limit)
        for key, limit in CORE_HOOK_LIMITS.items()
    }

    raw_modifiers = raw.get("human_modifiers") or {}
    human_modifiers = {
        key: _clamp_int(raw_modifiers.get(key, 0), bounds[0], bounds[1])
        for key, bounds in HUMAN_MODIFIER_LIMITS.items()
    }
    if not HOOK_NATURALNESS_ENABLED:
        human_modifiers["naturalness"] = 0

    raw_penalties = raw.get("penalties") or raw.get("hook_penalties") or {}
    penalties = {
        key: _normalize_penalty(raw_penalties.get(key, 0), bounds[0])
        for key, bounds in PENALTY_LIMITS.items()
    }
    if not HOOK_FORCED_INTRO_PENALTY_ENABLED:
        for key in FORCED_PENALTY_KEYS:
            penalties[key] = 0

    base_score, final_score = calculate_hook_scores(
        core_scores,
        human_modifiers,
        penalties,
    )

    hook_type = str(raw.get("hook_type", "MIXED") or "MIXED").strip().upper()
    if hook_type not in HOOK_TYPES:
        hook_type = "MIXED"

    raw_secondary = raw.get("secondary_hook_types")
    if raw_secondary is None:
        single = raw.get("secondary_hook_type")
        raw_secondary = [single] if single else []
    secondary: list[str] = []
    for item in raw_secondary or []:
        value = str(item or "").strip().upper()
        if value in HOOK_TYPES and value != hook_type and value not in secondary:
            secondary.append(value)

    return HookStartCandidate(
        start=round(float(raw.get("start", 0.0)), 3),
        base_hook_score=base_score,
        final_hook_score=final_score,
        core_scores=core_scores,
        human_modifiers=human_modifiers,
        penalties=penalties,
        hook_type=hook_type,
        secondary_hook_types=secondary[:5],
        reason=str(raw.get("hook_reason") or raw.get("reason") or "").strip()[:280],
        confidence=_clamp_float(
            raw.get("hook_confidence", raw.get("confidence", 0.0)),
            0.0,
            1.0,
        ),
        loop_score=_clamp_int(raw.get("loop_score", 0), 0, 100),
        continuity_score=0.0,
    )


def _segment_start_score(start: float, transcript: list[dict]) -> float:
    score = 0.0
    for segment in transcript or []:
        try:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
        except (TypeError, ValueError):
            continue
        text = str(segment.get("text", "")).strip().lower()
        if abs(start - seg_start) <= 0.08:
            score += 2.0
        if seg_start < start < seg_end:
            score -= 0.35
            if text.startswith(
                (
                    "and ",
                    "but ",
                    "because ",
                    "so then ",
                    "like i was",
                    "as i said",
                    "as i mentioned",
                )
            ):
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
        visual_boundary_bonus = max(0.0, float(candidate.continuity_score or 0.0))
        candidate.continuity_score = (
            continuity_score(candidate.start, transcript, original_start)
            + visual_boundary_bonus
        )

    best_score = max(candidate.final_hook_score for candidate in candidates)
    close = [
        candidate
        for candidate in candidates
        if best_score - candidate.final_hook_score <= HOOK_SCORE_TIE_THRESHOLD
    ]

    # Tie order follows the spec: natural speech/continuity, naturalness,
    # context independence, lower forced-hook penalty, then confidence/score.
    return max(
        close,
        key=lambda candidate: (
            candidate.continuity_score,
            candidate.naturalness,
            candidate.core_scores.get("context_independence", 0),
            candidate.forced_hook_penalty,
            candidate.confidence,
            candidate.final_hook_score,
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
    """Align Gemini's semantic START to Whisper/speech boundaries and tiny pre-roll."""
    semantic_start = clamp_semantic_start(
        semantic_start,
        original_start,
        highlight_end,
    )
    words = flatten_words(transcript)
    if not words:
        return semantic_start

    aligned = snap_to_word_boundary(semantic_start, "start", words)
    boundaries = find_boundary_candidates(
        aligned,
        "start",
        transcript,
        words,
        search_window=max(0.35, HOOK_MICRO_PREROLL_MAX + 0.10),
    )
    viable = [
        item
        for item in boundaries
        if aligned - HOOK_MICRO_PREROLL_MAX <= item.time <= aligned + 0.12
    ]
    if viable:
        best = max(
            viable,
            key=lambda item: item.score - abs(item.time - aligned) * 0.45,
        )
        adjusted = best.time
    else:
        adjusted = aligned

    min_preroll = max(0.0, float(HOOK_MICRO_PREROLL_MIN))
    max_preroll = max(min_preroll, float(HOOK_MICRO_PREROLL_MAX))
    preroll = min(max_preroll, min_preroll)
    if preroll > 0:
        proposed = max(0.0, adjusted - preroll)
        cuts_word = any(
            word.start + 0.015 < proposed < word.end - 0.015
            for word in words
        )
        if not cuts_word:
            adjusted = proposed

    return clamp_semantic_start(
        adjusted,
        original_start,
        highlight_end,
    )


def profile_priority_text(profile: str | None) -> str:
    return PROFILE_PRIORITIES.get(
        str(profile or "auto").strip().lower(),
        PROFILE_PRIORITIES["general"],
    )
