from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ollama

from src.config import (
    HOOK_CACHE_DIR,
    HOOK_OPTIMIZER_VERSION,
    HOOK_SEARCH_BEFORE,
    OLLAMA_MODEL,
)
from src.highlights.gemini_judge import build_transcript_context
from src.hooks.compact import (
    COMPACT_TOP_K,
    build_local_compact_fallback,
    reconstruct_compact_candidate,
)
from src.hooks.scoring import profile_priority_text
from src.logger import info, warning


_QWEN_HOOK_VERSION = "hook-qwen-compact-v2"
_CORE_KEYS = (
    "immediate_action",
    "curiosity_gap",
    "emotional_reaction",
    "conflict_tension",
    "visual_surprise",
    "context_independence",
    "payoff_proximity",
)
_MODIFIER_KEYS = (
    "contradiction",
    "specificity",
    "timeframe_tension",
    "relatability",
    "naturalness",
)
_PENALTY_KEYS = (
    "dead_air",
    "context_dependency",
    "spoiled_payoff",
    "mid_sentence",
    "duplicate_information",
    "forced_hook",
    "generic_setup",
    "forced_intro",
    "youtuber_intro",
    "fake_hype",
    "obvious_clickbait",
    "repeated_context",
    "fake_urgency",
)


def _cache_key(
    *,
    video_path: Path,
    original_start: float,
    highlight_end: float,
    candidate_starts: list[float],
    profile: str,
) -> str:
    try:
        stat = video_path.stat()
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
    except OSError:
        size = 0
        mtime_ns = 0
    payload = {
        "video": str(video_path.resolve()).lower(),
        "size": size,
        "mtime_ns": mtime_ns,
        "highlight_start": round(original_start, 3),
        "highlight_end": round(highlight_end, 3),
        "candidate_starts": [round(value, 3) for value in candidate_starts],
        "profile": profile,
        "model": OLLAMA_MODEL,
        "optimizer_version": HOOK_OPTIMIZER_VERSION,
        "fallback_version": _QWEN_HOOK_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_path(key: str) -> Path:
    return HOOK_CACHE_DIR / "qwen" / f"{key}.json"


def _load_cache(key: str) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _save_cache(key: str, payload: dict) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _parse_json(content: str) -> dict:
    text = str(content or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Qwen Hook Optimizer JSON must be an object")
    return payload


def _nearest_allowed(value, candidate_starts: list[float]) -> float | None:
    try:
        reported = float(value)
    except (TypeError, ValueError):
        return None
    if not candidate_starts:
        return None
    nearest = min(candidate_starts, key=lambda item: abs(float(item) - reported))
    if abs(float(nearest) - reported) > 0.12:
        return None
    return round(float(nearest), 3)


def _sanitize_legacy_candidate(raw: dict, start: float) -> dict:
    item = dict(raw)
    item["start"] = start
    core = item.get("core_scores") if isinstance(item.get("core_scores"), dict) else {}
    modifiers = item.get("human_modifiers") if isinstance(item.get("human_modifiers"), dict) else {}
    penalties = item.get("penalties") if isinstance(item.get("penalties"), dict) else {}
    item["core_scores"] = {key: core.get(key, 0) for key in _CORE_KEYS}
    item["human_modifiers"] = {key: modifiers.get(key, 0) for key in _MODIFIER_KEYS}
    item["penalties"] = {key: penalties.get(key, 0) for key in _PENALTY_KEYS}
    item.setdefault("base_hook_score", 0)
    item.setdefault("final_hook_score", 0)
    item.setdefault("hook_type", "MIXED")
    item.setdefault("secondary_hook_types", [])
    item.setdefault("hook_reason", item.get("reason") or "Local Qwen hook analysis.")
    item.setdefault("hook_confidence", item.get("confidence", 0.55))
    item.setdefault("loop_score", 0)
    return item


def _sanitize_payload(
    payload: dict,
    candidate_starts: list[float],
    *,
    transcript: list[dict] | None = None,
    video_path: Path | None = None,
    original_start: float | None = None,
    highlight_end: float | None = None,
) -> dict:
    """Accept compact V2 output and expand it to the historical detailed contract.

    Legacy detailed payloads remain accepted for tests/debug tooling. Compact V2
    returns at most Top 3; Python restores detailed scores and local media signals.
    """
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Qwen Hook Optimizer JSON missing candidates")

    original_start = float(original_start if original_start is not None else (candidate_starts[0] if candidate_starts else 0.0))
    highlight_end = float(highlight_end if highlight_end is not None else original_start + 30.0)

    normalized_raw: list[tuple[dict, float, bool]] = []
    seen: set[float] = set()
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        start = _nearest_allowed(raw.get("start"), candidate_starts)
        if start is None or start in seen:
            continue
        seen.add(start)
        is_legacy = isinstance(raw.get("core_scores"), dict) or isinstance(raw.get("hook_scores"), dict)
        normalized_raw.append((raw, start, is_legacy))

    if not normalized_raw:
        raise ValueError("Qwen Hook Optimizer returned no usable candidates")

    compact_mode = any(not item[2] for item in normalized_raw)
    if compact_mode:
        normalized_raw.sort(
            key=lambda item: (
                float(item[0].get("hook_score", item[0].get("score", 0)) or 0),
                float(item[0].get("confidence", 0) or 0),
                -abs(item[1] - original_start),
            ),
            reverse=True,
        )
        normalized_raw = normalized_raw[:COMPACT_TOP_K]

    sanitized: list[dict] = []
    for raw, start, is_legacy in normalized_raw:
        if is_legacy:
            item = _sanitize_legacy_candidate(raw, start)
        else:
            compact_raw = dict(raw)
            compact_raw["start"] = start
            item = reconstruct_compact_candidate(
                compact_raw,
                transcript=transcript,
                video_path=video_path,
                original_start=original_start,
                highlight_end=highlight_end,
            )
        sanitized.append(item)

    allowed_returned = {round(float(item["start"]), 3) for item in sanitized}
    best_start = _nearest_allowed(payload.get("best_start"), candidate_starts)
    if best_start not in allowed_returned:
        best_item = max(
            sanitized,
            key=lambda item: (
                float(item.get("_compact_semantic_score", item.get("final_hook_score", 0)) or 0),
                float(item.get("hook_confidence", 0) or 0),
                -abs(float(item["start"]) - original_start),
            ),
        )
        best_start = round(float(best_item["start"]), 3)

    sanitized.sort(key=lambda item: float(item["start"]))
    result = dict(payload)
    result["candidates"] = sanitized
    result["best_start"] = best_start
    result["_source"] = "qwen_local_fallback"
    result["_mode"] = "compact_v2" if compact_mode else "legacy_detailed_compat"
    return result


def build_qwen_hook_prompt(
    *,
    original_start: float,
    highlight_end: float,
    candidate_starts: list[float],
    transcript_text: str,
    profile: str,
    context_start: float,
    context_end: float,
) -> str:
    return f"""
You are the LOCAL QWEN semantic selector for HOOK START V2 COMPACT.
The highlight is already selected. Evaluate all supplied START timestamps internally,
but RETURN ONLY the TOP {COMPACT_TOP_K} semantic starts. Python will calculate the detailed
rubric, audio onset, frame boundary and reaction signals after your response.

CONTENT PROFILE: {profile}
PROFILE PRIORITIES: {profile_priority_text(profile)}
ORIGINAL HIGHLIGHT: {original_start:.3f}s -> {highlight_end:.3f}s
AVAILABLE CONTEXT: {context_start:.3f}s -> {context_end:.3f}s
CANDIDATE STARTS: {json.dumps(candidate_starts, separators=(',', ':'))}

Judge semantic first-second retention using:
- immediate action or natural reaction
- curiosity gap / unresolved tension
- minimum context required to understand the moment
- natural speech start; avoid mid-thought or generic setup
- nearby payoff without starting after/spoiling it
- specificity and authenticity over fake hype

FOUR-QUESTION CHECK:
1. What is happening?
2. Why should the viewer care?
3. What happens next?
4. Does the opening feel natural/real?

IMPORTANT:
- You have transcript context, NOT video vision. Do not invent visual facts.
- Evaluate every supplied timestamp internally, but DO NOT output all of them.
- Return at most {COMPACT_TOP_K} candidates, strongest first.
- Each reason must be one short sentence, max 18 words.
- best_start MUST exactly match one returned candidate.
- Never invent a timestamp outside CANDIDATE STARTS.

TIMESTAMPED TRANSCRIPT:
{transcript_text or "(no dialogue in this context)"}

Return ONLY strict JSON in exactly this compact shape:
{{
  "candidates": [
    {{"start": 0.0, "hook_score": 0, "confidence": 0.0, "reason": "short reason"}}
  ],
  "best_start": 0.0
}}
""".strip()


class QwenHookAnalyzer:
    def analyze(
        self,
        *,
        video_path: Path,
        transcript: list[dict],
        original_start: float,
        highlight_end: float,
        candidate_starts: list[float],
        profile: str,
        video_duration: float,
        clip_index: int,
    ) -> tuple[dict, bool]:
        if not candidate_starts:
            raise ValueError("Qwen Hook Optimizer received no candidate starts")

        key = _cache_key(
            video_path=video_path,
            original_start=original_start,
            highlight_end=highlight_end,
            candidate_starts=candidate_starts,
            profile=profile,
        )
        cached = _load_cache(key)
        if cached is not None:
            info(f"[HOOK COMPACT {clip_index}] Qwen cache hit")
            return cached, True

        context_start = max(0.0, original_start - HOOK_SEARCH_BEFORE - 0.50)
        context_end = min(float(video_duration), highlight_end + 0.50)
        transcript_text = build_transcript_context(
            transcript,
            context_start,
            context_end,
        )
        prompt = build_qwen_hook_prompt(
            original_start=original_start,
            highlight_end=highlight_end,
            candidate_starts=candidate_starts,
            transcript_text=transcript_text,
            profile=profile,
            context_start=context_start,
            context_end=context_end,
        )

        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    stream=False,
                    think=False,
                    keep_alive="30m",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return strict compact JSON only. Select at most three hook starts; "
                                "do not emit the detailed scoring rubric."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    format="json",
                    options={
                        "temperature": 0.08 if attempt == 1 else 0.0,
                        "num_ctx": 4096,
                        "num_predict": 640,
                    },
                )
                payload = _parse_json(response["message"]["content"])
                payload = _sanitize_payload(
                    payload,
                    candidate_starts,
                    transcript=transcript,
                    video_path=video_path,
                    original_start=original_start,
                    highlight_end=highlight_end,
                )
                _save_cache(key, payload)
                info(
                    f"[HOOK COMPACT {clip_index}] Qwen Top-{len(payload['candidates'])} "
                    "expanded locally with audio/frame signals"
                )
                return payload, False
            except Exception as exc:
                last_error = exc
                warning(f"[HOOK COMPACT {clip_index}] attempt={attempt} failed: {exc}")

        fallback = build_local_compact_fallback(
            candidate_starts=candidate_starts,
            transcript=transcript,
            video_path=video_path,
            original_start=original_start,
            highlight_end=highlight_end,
        )
        fallback["_qwen_error"] = str(last_error or "unknown compact Qwen failure")[:300]
        _save_cache(key, fallback)
        warning(
            f"[HOOK COMPACT {clip_index}] Qwen failed twice; using deterministic local Top-3 fallback"
        )
        return fallback, False
