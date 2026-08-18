from __future__ import annotations

from pathlib import Path
from typing import Any

from src.hooks.local_refinement import (
    _audio_boundary_score,
    _extract_audio,
    _visual_and_reaction_score,
)
from src.hooks.scoring import (
    CORE_HOOK_LIMITS,
    HUMAN_MODIFIER_LIMITS,
    PENALTY_LIMITS,
    continuity_score,
)
from src.smart_cut import flatten_words


COMPACT_TOP_K = 3

_GENERIC_SETUP = (
    "hey guys",
    "what's up guys",
    "whats up guys",
    "welcome back",
    "today we're",
    "today we are",
    "in this video",
)
_CONNECTIVE_STARTS = (
    "and ",
    "but ",
    "because ",
    "so then ",
    "as i said",
    "as i mentioned",
)
_FAKE_HYPE = ("insane", "crazy", "unbelievable", "you won't believe", "you wont believe")
_REACTION_WORDS = ("wait", "what", "oh", "no", "wow", "why", "how", "help", "run")
_CONFLICT_WORDS = ("fight", "attack", "enemy", "die", "dead", "kill", "run", "danger", "wrong")
_ACTION_WORDS = ("go", "run", "move", "jump", "shoot", "hit", "open", "close", "take", "get")


def _clamp(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clamp_int(value: Any, minimum: int, maximum: int, default: int = 0) -> int:
    return int(round(_clamp(value, minimum, maximum, default)))


def _segment_at(transcript: list[dict], start: float) -> tuple[dict | None, str]:
    best: dict | None = None
    best_distance = float("inf")
    for segment in transcript or []:
        try:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
        except (TypeError, ValueError):
            continue
        if seg_start <= start <= seg_end:
            return segment, str(segment.get("text", "") or "").strip()
        distance = min(abs(start - seg_start), abs(start - seg_end))
        if distance < best_distance:
            best = segment
            best_distance = distance
    return best, str((best or {}).get("text", "") or "").strip()


def _inside_word(transcript: list[dict], start: float) -> bool:
    for word in flatten_words(transcript):
        if word.start + 0.015 < start < word.end - 0.015:
            return True
    return False


def _local_media_signals(video_path: Path | None, start: float) -> dict[str, float]:
    if video_path is None:
        return {"audio": 0.0, "visual": 0.0, "reaction": 0.0}
    try:
        path = Path(video_path)
    except TypeError:
        return {"audio": 0.0, "visual": 0.0, "reaction": 0.0}
    if not path.exists():
        return {"audio": 0.0, "visual": 0.0, "reaction": 0.0}

    signal, signal_start = _extract_audio(path, max(0.0, start - 0.18), start + 0.22)
    audio = _audio_boundary_score(signal, signal_start, start)
    visual, reaction = _visual_and_reaction_score(path, start)
    return {
        "audio": round(_clamp(audio, 0.0, 1.0), 4),
        "visual": round(_clamp(visual, 0.0, 1.0), 4),
        "reaction": round(_clamp(reaction, 0.0, 1.0), 4),
    }


def _infer_hook_type(text: str, reason: str) -> tuple[str, list[str]]:
    haystack = f"{text} {reason}".lower()
    primary = "MIXED"
    secondary: list[str] = []

    if "?" in text or any(token in haystack.split()[:2] for token in ("why", "what", "how")):
        primary = "QUESTION"
    elif any(token in haystack for token in _CONFLICT_WORDS):
        primary = "CONFLICT"
    elif any(token in haystack for token in _REACTION_WORDS):
        primary = "REACTION"
    elif any(token in haystack for token in _ACTION_WORDS):
        primary = "ACTION"
    elif any(token in haystack for token in ("surprise", "unexpected", "reveal")):
        primary = "SURPRISE"

    for hook_type, needles in (
        ("REACTION", _REACTION_WORDS),
        ("CONFLICT", _CONFLICT_WORDS),
        ("ACTION", _ACTION_WORDS),
        ("SURPRISE", ("surprise", "unexpected", "reveal")),
        ("CURIOSITY", ("why", "what", "how", "wait")),
    ):
        if hook_type != primary and any(token in haystack for token in needles):
            secondary.append(hook_type)
    return primary, secondary[:3]


def reconstruct_compact_candidate(
    raw: dict,
    *,
    transcript: list[dict] | None,
    video_path: Path | None,
    original_start: float,
    highlight_end: float,
) -> dict:
    """Expand one compact Qwen Top-3 choice into the legacy detailed scoring contract.

    Qwen owns the semantic ranking only. Python deterministically reconstructs the
    detailed rubric and adds weak audio/frame/reaction evidence. The downstream
    optimizer remains the final authority and still performs its exact-boundary
    multimodal refinement.
    """
    transcript = list(transcript or [])
    start = float(raw.get("start", original_start))
    semantic_score = _clamp_int(raw.get("hook_score", raw.get("score", 65)), 0, 100, 65)
    confidence = _clamp(raw.get("confidence", 0.65), 0.0, 1.0, 0.65)
    reason = " ".join(str(raw.get("reason") or "Compact Qwen semantic selection.").split())[:220]

    segment, text = _segment_at(transcript, start)
    text_lower = text.lower()
    seg_start = None
    seg_end = None
    if segment is not None:
        try:
            seg_start = float(segment.get("start"))
            seg_end = float(segment.get("end"))
        except (TypeError, ValueError):
            seg_start = seg_end = None

    boundary = continuity_score(start, transcript, original_start) if transcript else 0.0
    inside_word = _inside_word(transcript, start) if transcript else False
    media = _local_media_signals(video_path, start)

    shift = start - float(original_start)
    available = max(0.5, float(highlight_end) - float(original_start))
    payoff_nearness = _clamp(1.0 - max(0.0, float(highlight_end) - start) / available, 0.0, 1.0)

    # Distribute the compact semantic score into the historical 0-100 rubric.
    # The fractions deliberately sum below 1.0; local modifiers complete the
    # score without allowing weak machine signals to overpower Qwen semantics.
    core_scores = {
        "immediate_action": _clamp_int(semantic_score * 0.19 + media["audio"] * 2.0 + media["visual"] * 1.5, 0, 20),
        "curiosity_gap": _clamp_int(semantic_score * 0.20, 0, 20),
        "emotional_reaction": _clamp_int(semantic_score * 0.12 + media["reaction"] * 2.0, 0, 15),
        "conflict_tension": _clamp_int(semantic_score * 0.12, 0, 15),
        "visual_surprise": _clamp_int(semantic_score * 0.035 + media["visual"] * 3.0, 0, 10),
        "context_independence": _clamp_int(semantic_score * 0.10 + max(-1.0, min(2.0, boundary)), 0, 10),
        "payoff_proximity": _clamp_int(semantic_score * 0.10 + payoff_nearness * 2.0, 0, 10),
    }

    if any(token in text_lower for token in _ACTION_WORDS):
        core_scores["immediate_action"] = min(20, core_scores["immediate_action"] + 2)
    if any(token in text_lower for token in _REACTION_WORDS):
        core_scores["emotional_reaction"] = min(15, core_scores["emotional_reaction"] + 2)
        core_scores["curiosity_gap"] = min(20, core_scores["curiosity_gap"] + 1)
    if any(token in text_lower for token in _CONFLICT_WORDS):
        core_scores["conflict_tension"] = min(15, core_scores["conflict_tension"] + 2)

    naturalness = _clamp_int(4 + boundary * 1.5 + media["audio"] * 2.0 + media["reaction"], -15, 10, 4)
    specificity = 2
    if any(char.isdigit() for char in text):
        specificity += 1
    if len(text.split()) >= 5:
        specificity += 1
    if len(text.split()) >= 10:
        specificity += 1

    human_modifiers = {
        "contradiction": 2 if any(token in text_lower for token in ("but ", "actually", "wait")) else 0,
        "specificity": _clamp_int(specificity, *HUMAN_MODIFIER_LIMITS["specificity"]),
        "timeframe_tension": 2 if any(token in text_lower for token in ("now", "before", "seconds", "time")) else 0,
        "relatability": 2 if text else 0,
        "naturalness": naturalness,
    }

    penalties = {key: 0 for key in PENALTY_LIMITS}
    if inside_word:
        penalties["mid_sentence"] = -12
    elif seg_start is not None and seg_end is not None and seg_start + 0.22 < start < seg_end - 0.08:
        penalties["mid_sentence"] = -4
    if text_lower.startswith(_CONNECTIVE_STARTS):
        penalties["context_dependency"] = -6
    if any(phrase in text_lower for phrase in _GENERIC_SETUP):
        penalties["generic_setup"] = -10
        penalties["forced_intro"] = -8
        if "guys" in text_lower or "welcome back" in text_lower:
            penalties["youtuber_intro"] = -12
    if any(phrase in text_lower for phrase in _FAKE_HYPE):
        penalties["fake_hype"] = -5
    if not text and media["audio"] < 0.18:
        penalties["dead_air"] = -8
    if shift > 2.5 and payoff_nearness > 0.75:
        penalties["spoiled_payoff"] = -4

    hook_type, secondary = _infer_hook_type(text, reason)
    loop_score = _clamp_int(semantic_score * 0.75 + confidence * 15.0, 0, 100)

    return {
        "start": round(start, 3),
        "core_scores": core_scores,
        "human_modifiers": human_modifiers,
        "penalties": penalties,
        "hook_type": hook_type,
        "secondary_hook_types": secondary,
        "hook_reason": reason,
        "hook_confidence": round(confidence, 3),
        "loop_score": loop_score,
        "base_hook_score": semantic_score,
        "final_hook_score": semantic_score,
        "_compact_semantic_score": semantic_score,
        "_compact_local_signals": media,
        "_compact_boundary_score": round(boundary, 4),
    }


def build_local_compact_fallback(
    *,
    candidate_starts: list[float],
    transcript: list[dict] | None,
    video_path: Path | None,
    original_start: float,
    highlight_end: float,
    top_k: int = COMPACT_TOP_K,
) -> dict:
    """Deterministic no-LLM fallback: rank all starts cheaply, then expand Top 3."""
    transcript = list(transcript or [])
    compact: list[dict] = []
    for start in candidate_starts:
        boundary = continuity_score(start, transcript, original_start) if transcript else 0.0
        _, text = _segment_at(transcript, start)
        text_lower = text.lower()
        score = 70.0 - min(14.0, abs(start - original_start) * 3.0)
        score += max(-8.0, min(8.0, boundary * 2.0))
        if any(token in text_lower for token in _REACTION_WORDS):
            score += 4.0
        if any(token in text_lower for token in _ACTION_WORDS):
            score += 3.0
        if any(phrase in text_lower for phrase in _GENERIC_SETUP):
            score -= 8.0
        compact.append(
            {
                "start": round(float(start), 3),
                "hook_score": _clamp_int(score, 0, 100),
                "confidence": 0.52,
                "reason": "Deterministic local fallback after compact Qwen failure.",
            }
        )

    compact.sort(key=lambda item: (item["hook_score"], -abs(item["start"] - original_start)), reverse=True)
    selected = compact[: max(1, min(int(top_k), len(compact)))]
    detailed = [
        reconstruct_compact_candidate(
            item,
            transcript=transcript,
            video_path=video_path,
            original_start=original_start,
            highlight_end=highlight_end,
        )
        for item in selected
    ]
    return {
        "candidates": detailed,
        "best_start": detailed[0]["start"] if detailed else round(float(original_start), 3),
        "_source": "qwen_local_fallback",
        "_mode": "compact_v2_local_fallback",
    }
