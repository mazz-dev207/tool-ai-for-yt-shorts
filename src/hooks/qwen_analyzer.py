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
from src.hooks.scoring import profile_priority_text
from src.logger import info


_QWEN_HOOK_VERSION = "hook-qwen-fallback-v1"
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


def _sanitize_payload(payload: dict, candidate_starts: list[float]) -> dict:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Qwen Hook Optimizer JSON missing candidates")

    sanitized: list[dict] = []
    seen: set[float] = set()
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        try:
            reported_start = float(raw.get("start"))
        except (TypeError, ValueError):
            continue
        nearest = min(candidate_starts, key=lambda value: abs(value - reported_start))
        if abs(nearest - reported_start) > 0.12:
            continue
        normalized_start = round(nearest, 3)
        if normalized_start in seen:
            continue

        item = dict(raw)
        item["start"] = normalized_start
        core = item.get("core_scores") if isinstance(item.get("core_scores"), dict) else {}
        modifiers = (
            item.get("human_modifiers")
            if isinstance(item.get("human_modifiers"), dict)
            else {}
        )
        penalties = item.get("penalties") if isinstance(item.get("penalties"), dict) else {}
        item["core_scores"] = {key: core.get(key, 0) for key in _CORE_KEYS}
        item["human_modifiers"] = {key: modifiers.get(key, 0) for key in _MODIFIER_KEYS}
        item["penalties"] = {key: penalties.get(key, 0) for key in _PENALTY_KEYS}
        item.setdefault("base_hook_score", 0)
        item.setdefault("final_hook_score", 0)
        item.setdefault("hook_type", "MIXED")
        item.setdefault("secondary_hook_types", [])
        item.setdefault("hook_reason", "Local Qwen fallback analysis.")
        item.setdefault("hook_confidence", 0.55)
        item.setdefault("loop_score", 0)
        sanitized.append(item)
        seen.add(normalized_start)

    if not sanitized:
        raise ValueError("Qwen Hook Optimizer returned no usable candidates")

    sanitized.sort(key=lambda item: float(item["start"]))
    payload = dict(payload)
    payload["candidates"] = sanitized
    if not isinstance(payload.get("best_start"), (int, float)):
        payload["best_start"] = sanitized[0]["start"]
    payload["_source"] = "qwen_local_fallback"
    return payload


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
You are the LOCAL QWEN FALLBACK for a YouTube Shorts Hook START Optimizer.
Gemini multimodal analysis is unavailable. The highlight is already selected.
Choose the strongest START timestamp from the supplied candidates for first-second retention.

IMPORTANT LIMITATION:
You have TIMESTAMPED TRANSCRIPT context, not direct video vision. Be conservative about visual claims.
Set visual_surprise low (normally 0-3) unless the transcript clearly describes a visual surprise.
The local pipeline will independently add frame-boundary, audio-onset and reaction refinement after your semantic choice.

CONTENT PROFILE: {profile}
PROFILE PRIORITIES: {profile_priority_text(profile)}
ORIGINAL HIGHLIGHT: {original_start:.3f}s -> {highlight_end:.3f}s
AVAILABLE CONTEXT: {context_start:.3f}s -> {context_end:.3f}s
CANDIDATE STARTS: {json.dumps(candidate_starts, separators=(',', ':'))}

For EVERY candidate return these fields:
- start
- core_scores: immediate_action 0-20, curiosity_gap 0-20, emotional_reaction 0-15,
  conflict_tension 0-15, visual_surprise 0-10, context_independence 0-10, payoff_proximity 0-10
- human_modifiers: contradiction -5..8, specificity 0..6, timeframe_tension 0..5,
  relatability 0..5, naturalness -15..10
- penalties as NEGATIVE values or zero: dead_air, context_dependency, spoiled_payoff,
  mid_sentence, duplicate_information, forced_hook, generic_setup, forced_intro,
  youtuber_intro, fake_hype, obvious_clickbait, repeated_context, fake_urgency
- hook_type
- secondary_hook_types
- hook_reason: one short sentence
- hook_confidence: 0.0-1.0
- loop_score: 0-100
- base_hook_score and final_hook_score may be estimates; local code recalculates them.

Prefer unresolved tension, natural reaction, immediate action, a clean question in the viewer's mind,
and the minimum context needed before a nearby payoff. Penalize starts that begin mid-thought,
repeat missing context, start after the payoff, feel like generic setup, or require information before the clip.
Do not reward hype words by themselves. Naturalness and context independence matter more than loud wording.

TIMESTAMPED TRANSCRIPT:
{transcript_text or "(no dialogue in this context)"}

Return ONLY JSON with this shape:
{{"candidates":[...],"best_start":0.0}}
Return each supplied candidate at most once and do not invent timestamps outside the list.
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
            info(f"[HOOK LOCAL {clip_index}] Qwen cache hit")
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
                            "content": "Return strict JSON only. Score hook starts conservatively and truthfully.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    format="json",
                    options={
                        "temperature": 0.10 if attempt == 1 else 0.0,
                        "num_ctx": 4096,
                        "num_predict": 1800,
                    },
                )
                payload = _parse_json(response["message"]["content"])
                payload = _sanitize_payload(payload, candidate_starts)
                _save_cache(key, payload)
                info(
                    f"[HOOK LOCAL {clip_index}] Qwen fallback produced "
                    f"{len(payload['candidates'])} candidates"
                )
                return payload, False
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Qwen Hook Optimizer failed after retry: {last_error}")
