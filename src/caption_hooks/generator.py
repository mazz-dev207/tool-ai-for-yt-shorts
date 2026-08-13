from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import ollama

from src.caption_hooks.models import CaptionHookCandidate, CaptionHookResult
from src.caption_hooks.scorer import (
    choose_emphasis,
    normalize_caption_text,
    score_caption_candidate,
    split_caption_lines,
    tokens,
)
from src.config import OLLAMA_MODEL
from src.logger import info, warning
from src.v3_config import (
    V3_CAPTION_HOOK_ALLOW_NONE,
    V3_CAPTION_HOOK_CACHE_DIR,
    V3_CAPTION_HOOK_DEFAULT_DURATION,
    V3_CAPTION_HOOK_ENABLED,
    V3_CAPTION_HOOK_GENERATOR,
    V3_CAPTION_HOOK_MAX_DURATION,
    V3_CAPTION_HOOK_MAX_LINES,
    V3_CAPTION_HOOK_MAX_WORDS,
    V3_CAPTION_HOOK_MIN_DURATION,
    V3_CAPTION_HOOK_MIN_SCORE,
    V3_CAPTION_HOOK_PROMPT_VERSION,
)


_ALLOWED_MODES = {"COMPLEMENT", "REINFORCE", "CLARIFY", "NONE"}
_ALLOWED_TYPES = {
    "outcome_tease",
    "contradiction",
    "mystery",
    "stakes",
    "reaction_setup",
    "challenge",
    "specific_fact",
    "conflict",
    "unexpected_discovery",
    "payoff_setup",
    "clarification",
    "none",
}


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _plan_value(edit_plan: dict, key: str) -> str:
    return _safe_text(edit_plan.get(key, "")) if isinstance(edit_plan, dict) else ""


def _opening_text(transcript: list[dict], edit_plan: dict, clip: dict) -> str:
    timeline = edit_plan.get("timeline") if isinstance(edit_plan, dict) else None
    if isinstance(timeline, list) and timeline:
        first = timeline[0]
        start = float(first.get("source_start", clip.get("start", 0.0)) or 0.0)
        end = min(
            float(first.get("source_end", start + 3.0) or (start + 3.0)),
            start + 3.0,
        )
    else:
        start = float(clip.get("start", 0.0) or 0.0)
        end = min(float(clip.get("end", start + 3.0) or (start + 3.0)), start + 3.0)

    parts: list[str] = []
    for segment in transcript or []:
        try:
            seg_start = float(segment.get("start", 0.0) or 0.0)
            seg_end = float(segment.get("end", seg_start) or seg_start)
        except (TypeError, ValueError):
            continue
        if seg_end < start or seg_start > end:
            continue
        text = _safe_text(segment.get("text", ""))
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _satisfaction_payload(clip: dict) -> dict:
    value = clip.get("satisfaction") or {}
    return value if isinstance(value, dict) else {}


def _truth_context(
    *,
    clip: dict,
    edit_plan: dict,
    opening_text: str,
    satisfaction: dict,
) -> str:
    hook = edit_plan.get("hook") or {} if isinstance(edit_plan, dict) else {}
    if not isinstance(hook, dict):
        hook = {}
    editorial = clip.get("v3") or {}
    if not isinstance(editorial, dict):
        editorial = {}
    pieces = [
        opening_text,
        _safe_text(hook.get("text", "")),
        _plan_value(edit_plan, "viewer_question"),
        _plan_value(edit_plan, "stakes"),
        _plan_value(edit_plan, "payoff"),
        _safe_text(clip.get("hook", "")),
        _safe_text(clip.get("payoff", "")),
        _safe_text(clip.get("title", "")),
        _safe_text(satisfaction.get("hook_promise", "")),
        _safe_text(satisfaction.get("actual_payoff", "")),
        _safe_text((satisfaction.get("payoff") or {}).get("reason", ""))
        if isinstance(satisfaction.get("payoff"), dict)
        else "",
        _safe_text(editorial.get("editorial_angle", "")),
        _safe_text(editorial.get("stakes", "")),
        _safe_text(editorial.get("payoff", "")),
    ]
    return " | ".join(item for item in pieces if item)


def _payoff_text(clip: dict, edit_plan: dict, satisfaction: dict) -> str:
    for value in (
        satisfaction.get("actual_payoff"),
        (satisfaction.get("payoff") or {}).get("reason")
        if isinstance(satisfaction.get("payoff"), dict)
        else "",
        edit_plan.get("payoff") if isinstance(edit_plan, dict) else "",
        clip.get("payoff"),
    ):
        text = _safe_text(value)
        if text:
            return text
    return ""


def _hook_score(clip: dict, edit_plan: dict) -> float:
    hook = edit_plan.get("hook") or {} if isinstance(edit_plan, dict) else {}
    for value in (
        hook.get("score") if isinstance(hook, dict) else None,
        (clip.get("v3") or {}).get("hook_score_v3")
        if isinstance(clip.get("v3"), dict)
        else None,
        clip.get("hook_score"),
        clip.get("base_hook_score"),
    ):
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _position_for_runtime() -> str:
    mode = os.getenv("SMARTCROP_MODE_RUNTIME", "auto").strip().lower()
    if mode == "gameplay-webcam":
        return "gameplay_top"
    return "top_safe"


def _duration_for(text: str) -> float:
    count = max(1, len(tokens(text)))
    target = 0.72 + count * 0.14
    if count <= 3:
        target = max(target, V3_CAPTION_HOOK_DEFAULT_DURATION - 0.2)
    return round(
        max(
            V3_CAPTION_HOOK_MIN_DURATION,
            min(V3_CAPTION_HOOK_MAX_DURATION, target),
        ),
        2,
    )


def _cache_key(
    *,
    video_name: str,
    clip_index: int,
    clip: dict,
    edit_plan: dict,
    content_profile: str,
    truth_context: str,
) -> str:
    payload = {
        "video_name": video_name,
        "clip_index": clip_index,
        "range": [clip.get("start"), clip.get("end")],
        "profile": content_profile,
        "hook": edit_plan.get("hook") if isinstance(edit_plan, dict) else None,
        "payoff": edit_plan.get("payoff") if isinstance(edit_plan, dict) else None,
        "truth": truth_context,
        "prompt_version": V3_CAPTION_HOOK_PROMPT_VERSION,
        "min_score": V3_CAPTION_HOOK_MIN_SCORE,
        "max_words": V3_CAPTION_HOOK_MAX_WORDS,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_path(key: str) -> Path:
    return V3_CAPTION_HOOK_CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> CaptionHookResult | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = CaptionHookResult.from_dict(payload)
        result.cache_hit = True
        result.source = f"{result.source}:cache"
        return result
    except Exception:
        return None


def _write_cache(key: str, result: CaptionHookResult) -> None:
    try:
        V3_CAPTION_HOOK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(key).write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        warning(f"[CAPTION HOOK] cache write failed: {exc}")


def _profile_guidance(profile: str) -> str:
    profile = str(profile or "general").lower()
    if profile == "gaming":
        return (
            "Prefer stakes, clutch/fail tension, challenge, unexpected discovery, "
            "specific score differences, reaction setup or impossible-looking action."
        )
    if profile == "entertainment":
        return (
            "Prefer challenge, conflict, reveal, twist, social tension, reaction or "
            "a truthful outcome tease."
        )
    if profile == "reaction":
        return "Prefer trigger, anticipation, realization, reaction setup or clarification."
    if profile in {"podcast", "interview"}:
        return (
            "Use a calmer editorial voice. Prefer specific claims, contradictions, "
            "key insights or surprising facts. Avoid gaming-style hype."
        )
    return "Prefer a concrete, self-contained reason to keep watching."


def _local_prompt(
    *,
    content_profile: str,
    opening_text: str,
    hook_text: str,
    stakes: str,
    payoff_text: str,
    truth_context: str,
    satisfaction: dict,
) -> tuple[str, str]:
    system = f'''You are a senior short-form video editor writing professional on-screen opening caption hooks.

Return ONLY JSON:
{{
  "candidates": [
    {{
      "text": "3 TO 7 WORD CAPTION",
      "type": "outcome_tease|contradiction|mystery|stakes|reaction_setup|challenge|specific_fact|conflict|unexpected_discovery|payoff_setup|clarification",
      "mode": "COMPLEMENT|REINFORCE|CLARIFY",
      "reason": "brief editorial reason",
      "truth_basis": "what supplied evidence supports it"
    }}
  ]
}}

Generate 3 to 5 genuinely different candidates.
Usually use 3-7 words; never exceed {V3_CAPTION_HOOK_MAX_WORDS} words.
Do not copy the opening dialogue.
Do not invent names, numbers, objects, scores, outcomes, reactions or events.
Do not use generic hype such as WATCH UNTIL THE END, YOU WON'T BELIEVE, THIS IS INSANE, or WHAT HAPPENS NEXT.
Do not spoil the exact resolution unless an outcome-first hook clearly creates a truthful how/why curiosity gap.
NONE is handled by downstream scoring, so only return candidates you can support from the evidence.
{_profile_guidance(content_profile)}'''.strip()
    risks = satisfaction.get("risks") or {}
    user = f'''CONTENT PROFILE: {content_profile}
OPENING DIALOGUE: {opening_text or "(none)"}
SELECTED HOOK: {hook_text or "(native / no text)"}
STAKES: {stakes or "(unknown)"}
ACTUAL PAYOFF: {payoff_text or "(unknown)"}
SATISFACTION STATUS: {satisfaction.get("status", "unknown")}
EXPECTATION MATCH: {satisfaction.get("expectation_match_score")}
CONTEXT INDEPENDENCE: {satisfaction.get("context_independence_score")}
RISKS: {json.dumps(risks, ensure_ascii=False)}

TRUTH EVIDENCE:
{truth_context}'''.strip()
    return system, user


def _parse_local_candidates(content: str) -> list[CaptionHookCandidate]:
    payload = json.loads(content)
    raw_items = payload.get("candidates", []) if isinstance(payload, dict) else []
    if not isinstance(raw_items, list):
        return []
    result: list[CaptionHookCandidate] = []
    seen: set[str] = set()
    for raw in raw_items[:5]:
        if not isinstance(raw, dict):
            continue
        text = normalize_caption_text(raw.get("text", ""))
        if not text or text in seen:
            continue
        mode = str(raw.get("mode", "COMPLEMENT") or "COMPLEMENT").upper()
        hook_type = str(raw.get("type", "none") or "none").lower()
        if mode not in _ALLOWED_MODES - {"NONE"}:
            mode = "COMPLEMENT"
        if hook_type not in _ALLOWED_TYPES - {"none"}:
            hook_type = "clarification"
        seen.add(text)
        result.append(
            CaptionHookCandidate(
                text=text,
                type=hook_type,
                mode=mode,
                reason=_safe_text(raw.get("reason", "")),
                truth_basis=_safe_text(raw.get("truth_basis", "")),
            )
        )
    return result


def _generate_local_candidates(
    *,
    content_profile: str,
    opening_text: str,
    edit_plan: dict,
    payoff_text: str,
    truth_context: str,
    satisfaction: dict,
) -> list[CaptionHookCandidate]:
    hook = edit_plan.get("hook") or {} if isinstance(edit_plan, dict) else {}
    hook_text = _safe_text(hook.get("text", "")) if isinstance(hook, dict) else ""
    system, user = _local_prompt(
        content_profile=content_profile,
        opening_text=opening_text,
        hook_text=hook_text,
        stakes=_plan_value(edit_plan, "stakes"),
        payoff_text=payoff_text,
        truth_context=truth_context,
        satisfaction=satisfaction,
    )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        stream=False,
        think=False,
        keep_alive="30m",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        options={
            "temperature": 0.20,
            "num_ctx": 4096,
            "num_predict": 650,
        },
    )
    return _parse_local_candidates(response["message"]["content"].strip())


def _seed_type(source: str, text: str) -> tuple[str, str]:
    source = source.lower()
    upper = normalize_caption_text(text)
    if source == "stakes":
        return "stakes", "COMPLEMENT"
    if source == "payoff":
        return "payoff_setup", "COMPLEMENT"
    if source == "hook":
        return "clarification", "REINFORCE"
    if any(char.isdigit() for char in upper):
        return "specific_fact", "COMPLEMENT"
    if source == "title":
        return "clarification", "CLARIFY"
    return "mystery", "COMPLEMENT"


def _deterministic_candidates(
    *,
    clip: dict,
    edit_plan: dict,
    satisfaction: dict,
) -> list[CaptionHookCandidate]:
    sources = [
        ("stakes", edit_plan.get("stakes", "") if isinstance(edit_plan, dict) else ""),
        (
            "hook",
            (edit_plan.get("hook") or {}).get("text", "")
            if isinstance(edit_plan, dict) and isinstance(edit_plan.get("hook"), dict)
            else "",
        ),
        ("payoff", satisfaction.get("hook_promise", "")),
        ("title", clip.get("title", "")),
    ]
    result: list[CaptionHookCandidate] = []
    seen: set[str] = set()
    for source, value in sources:
        text = normalize_caption_text(value)
        word_count = len(tokens(text))
        if not text or word_count < 2 or word_count > V3_CAPTION_HOOK_MAX_WORDS:
            continue
        if text in seen:
            continue
        hook_type, mode = _seed_type(source, text)
        result.append(
            CaptionHookCandidate(
                text=text,
                type=hook_type,
                mode=mode,
                reason=f"Conservative extraction from existing {source} evidence.",
                truth_basis=_safe_text(value),
            )
        )
        seen.add(text)
    return result[:5]


def _none_result(
    reason: str,
    *,
    source: str,
    candidates: list[CaptionHookCandidate],
) -> CaptionHookResult:
    return CaptionHookResult(
        enabled=False,
        status="none",
        mode="NONE",
        type="none",
        text="",
        score=max((item.score for item in candidates), default=None),
        reason=reason,
        source=source,
        candidates=[item.to_dict() for item in candidates],
    )


def generate_caption_hook(
    *,
    video_name: str,
    clip_index: int,
    clip: dict,
    transcript: list[dict],
    edit_plan: dict,
    content_profile: str,
) -> CaptionHookResult:
    if not V3_CAPTION_HOOK_ENABLED or V3_CAPTION_HOOK_GENERATOR == "off":
        return CaptionHookResult(
            enabled=False,
            status="disabled",
            reason="Caption Hook Engine disabled by configuration.",
        )

    satisfaction = _satisfaction_payload(clip)
    opening_text = _opening_text(transcript, edit_plan, clip)
    payoff_text = _payoff_text(clip, edit_plan, satisfaction)
    truth_context = _truth_context(
        clip=clip,
        edit_plan=edit_plan,
        opening_text=opening_text,
        satisfaction=satisfaction,
    )

    key = _cache_key(
        video_name=video_name,
        clip_index=clip_index,
        clip=clip,
        edit_plan=edit_plan,
        content_profile=content_profile,
        truth_context=truth_context,
    )
    cached = _read_cache(key)
    if cached is not None:
        info(
            f'[CAPTION HOOK] clip={clip_index} cache hit '
            f'enabled={cached.enabled} text="{cached.text}"'
        )
        return cached

    info(f"[CAPTION HOOK] clip={clip_index} generating candidates")
    candidates: list[CaptionHookCandidate] = []
    source = "deterministic"
    local_error: Exception | None = None

    if V3_CAPTION_HOOK_GENERATOR in {"auto", "local"}:
        try:
            candidates = _generate_local_candidates(
                content_profile=content_profile,
                opening_text=opening_text,
                edit_plan=edit_plan,
                payoff_text=payoff_text,
                truth_context=truth_context,
                satisfaction=satisfaction,
            )
            source = "ollama"
        except Exception as exc:
            local_error = exc
            warning(
                f"[CAPTION HOOK] clip={clip_index} local generation unavailable: {exc}"
            )

    if not candidates:
        candidates = _deterministic_candidates(
            clip=clip,
            edit_plan=edit_plan,
            satisfaction=satisfaction,
        )
        source = "deterministic"

    scored: list[CaptionHookCandidate] = []
    for candidate in candidates:
        scored_candidate = score_caption_candidate(
            candidate,
            opening_text=opening_text,
            payoff_text=payoff_text,
            truth_context=truth_context,
            satisfaction=satisfaction,
            content_profile=content_profile,
        )
        scored.append(scored_candidate)
        suffix = (
            f" rejected={','.join(scored_candidate.rejected_reasons)}"
            if scored_candidate.rejected_reasons
            else ""
        )
        info(
            f'[CAPTION HOOK] candidate="{scored_candidate.text}" '
            f"score={scored_candidate.score:.0f}{suffix}"
        )

    hard_reject = {"generic_hype", "clickbait", "unsupported_number"}
    eligible = [
        item
        for item in scored
        if not hard_reject.intersection(item.rejected_reasons)
    ]
    eligible.sort(key=lambda item: item.score, reverse=True)
    best = eligible[0] if eligible else None

    if best is None:
        if local_error is not None and not scored:
            result = CaptionHookResult(
                enabled=False,
                status="unavailable",
                mode="NONE",
                type="none",
                reason="Candidate generation unavailable; continuing without editorial caption.",
                source=source,
            )
        else:
            result = _none_result(
                "No truthful, non-clickbait caption candidate survived validation.",
                source=source,
                candidates=scored,
            )
        _write_cache(key, result)
        return result

    hook_score = _hook_score(clip, edit_plan)
    spoken_hook_is_strong = hook_score >= 92 and bool(opening_text)
    required_score = float(V3_CAPTION_HOOK_MIN_SCORE)
    if spoken_hook_is_strong:
        required_score = max(required_score, 85.0)

    if best.score < required_score and V3_CAPTION_HOOK_ALLOW_NONE:
        result = _none_result(
            (
                f"Best caption score {best.score:.0f} below required "
                f"{required_score:.0f}; spoken hook or clip works better without extra text."
            ),
            source=source,
            candidates=scored,
        )
        info(f"[CAPTION HOOK] clip={clip_index} NONE reason=score_below_threshold")
        _write_cache(key, result)
        return result

    lines = split_caption_lines(best.text, V3_CAPTION_HOOK_MAX_LINES)
    result = CaptionHookResult(
        enabled=True,
        status="ok",
        mode=best.mode,
        type=best.type,
        text=best.text,
        lines=lines,
        emphasis=choose_emphasis(best.text)[:1],
        start=0.05,
        duration=_duration_for(best.text),
        position=_position_for_runtime(),
        score=best.score,
        scores=best.scores,
        penalties=best.penalties,
        reason=best.reason or "Highest-scoring truthful caption candidate.",
        source=source,
        candidates=[item.to_dict() for item in scored],
    )
    info(f'[CAPTION HOOK] selected="{result.text}"')
    info(
        f"[CAPTION HOOK] type={result.type} mode={result.mode} "
        f"score={result.score:.0f} duration={result.duration:.2f}s "
        f"position={result.position}"
    )
    _write_cache(key, result)
    return result
