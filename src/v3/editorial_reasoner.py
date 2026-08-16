from __future__ import annotations

"""Local V3 editorial + Viewer Satisfaction reasoning.

Gemini is intentionally reserved for the upstream Final Multimodal Highlight
Judge. V3 Editorial and Viewer Satisfaction share ONE local Qwen/Ollama request
per attempt. Existing class/function names are preserved for compatibility with
the pipeline and regression tests.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import ollama

from src.config import OLLAMA_MODEL
from src.highlights.gemini_judge import build_transcript_context
from src.logger import info
from src.satisfaction.satisfaction_analyzer import build_end_candidate_values
from src.v3_config import (
    V3_CACHE_DIR,
    V3_CONTEXT_AFTER,
    V3_CONTEXT_BEFORE,
    V3_EDITORIAL_PROMPT_VERSION,
)


VIEWER_SATISFACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "viewer_satisfaction_score": {"type": "integer"},
        "payoff_score": {"type": "integer"},
        "expectation_match_score": {"type": "integer"},
        "context_independence_score": {"type": "integer"},
        "clarity_score": {"type": "integer"},
        "emotional_completeness_score": {"type": "integer"},
        "value_density_score": {"type": "integer"},
        "ending_quality_score": {"type": "integer"},
    },
}

EDITORIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "editorial_angle": {"type": "object"},
        "originality_analysis": {"type": "object"},
        "hook_candidates": {"type": "array"},
        "timeline": {"type": "array"},
        "context_overlays": {"type": "array"},
        "visual_events": {"type": "array"},
        "audio_events": {"type": "array"},
        "caption_emphasis": {"type": "array"},
        "recommended_transformations": {"type": "array"},
        "viewer_satisfaction": VIEWER_SATISFACTION_SCHEMA,
    },
    "required": [
        "editorial_angle",
        "originality_analysis",
        "hook_candidates",
        "timeline",
        "context_overlays",
        "visual_events",
        "audio_events",
        "caption_emphasis",
        "recommended_transformations",
    ],
}

_LOCAL_PROVIDER_VERSION = "v3-qwen-editorial-satisfaction-v1"
_LOCAL_CACHE_DIR = V3_CACHE_DIR / "qwen"
_LIST_LIMITS = {
    "hook_candidates": 6,
    "timeline": 8,
    "context_overlays": 4,
    "visual_events": 10,
    "audio_events": 6,
    "caption_emphasis": 8,
    "recommended_transformations": 10,
}


def _clamp_score(value: Any, default: int = 65) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(0, min(100, parsed))


def _clean(value: Any, limit: int = 320) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _parse_json(content: str) -> dict:
    text = str(content or "").strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        raw = json.loads(text[start : end + 1])
    if not isinstance(raw, dict):
        raise ValueError("V3 Qwen response must be a JSON object")
    return raw


def _multimodal_seed_scores(clip: dict) -> dict[str, int]:
    """Reuse the upstream Gemini Final Judge evidence without another API call."""
    gemini = clip.get("gemini") if isinstance(clip.get("gemini"), dict) else {}
    scores = gemini.get("scores") if isinstance(gemini.get("scores"), dict) else {}
    local_scores = clip.get("scores") if isinstance(clip.get("scores"), dict) else {}

    def read(*values, default=65):
        for value in values:
            if value is not None:
                return _clamp_score(value, default)
        return default

    return {
        "hook": read(scores.get("hook"), clip.get("hook_score"), local_scores.get("hook"), default=65),
        "payoff": read(scores.get("payoff"), local_scores.get("payoff"), default=65),
        "emotion": read(scores.get("emotion"), local_scores.get("emotional_intensity"), default=62),
        "visual_action": read(scores.get("visual_action"), default=55),
        "surprise": read(scores.get("surprise"), default=55),
        "standalone": read(scores.get("standalone"), local_scores.get("standalone_context"), default=65),
        "replayability": read(scores.get("replayability"), default=55),
    }


def _default_satisfaction(clip: dict, context_end: float) -> dict:
    """Safe local fallback if Qwen omits the Satisfaction block.

    It combines transcript-level downstream reasoning with the already-paid-for
    Gemini Final Judge evidence stored on the selected candidate. No API call is
    performed here.
    """
    seed = _multimodal_seed_scores(clip)
    gemini = clip.get("gemini") if isinstance(clip.get("gemini"), dict) else {}
    payoff_exists = bool(gemini.get("has_complete_payoff", seed["payoff"] >= 60))
    needs_context = bool(gemini.get("requires_previous_context", seed["standalone"] < 55))
    payoff = seed["payoff"]
    context = seed["standalone"]
    expectation = _clamp_score((seed["hook"] + payoff) / 2)
    clarity = _clamp_score((context * 0.7) + (seed["hook"] * 0.3))
    emotional = _clamp_score((seed["emotion"] * 0.75) + (payoff * 0.25))
    value_density = _clamp_score(
        (seed["hook"] + payoff + seed["visual_action"] + seed["surprise"]) / 4
    )
    ending_quality = _clamp_score((payoff * 0.70) + (emotional * 0.30))
    viewer = _clamp_score(
        payoff * 0.25
        + expectation * 0.18
        + context * 0.15
        + clarity * 0.12
        + emotional * 0.12
        + value_density * 0.10
        + ending_quality * 0.08
    )
    clip_end = float(clip.get("end", 0.0) or 0.0)
    allowed_ends = build_end_candidate_values(clip_end, context_end)
    reason = _clean(gemini.get("reason") or clip.get("payoff") or "Existing final-judge evidence.")
    hook_promise = _clean(clip.get("hook") or clip.get("title") or "Selected moment")
    actual_payoff = _clean(clip.get("payoff") or reason or "Selected payoff")

    return {
        "viewer_satisfaction_score": viewer,
        "payoff_score": payoff,
        "expectation_match_score": expectation,
        "context_independence_score": context,
        "clarity_score": clarity,
        "emotional_completeness_score": emotional,
        "value_density_score": value_density,
        "ending_quality_score": ending_quality,
        "hook_promise": hook_promise,
        "actual_payoff": actual_payoff,
        "structure": {
            "setup": True,
            "tension": seed["hook"] >= 60,
            "escalation": seed["emotion"] >= 55 or seed["visual_action"] >= 55,
            "payoff": payoff_exists,
            "pattern": "local_qwen_with_final_judge_evidence",
        },
        "payoff": {
            "exists": payoff_exists,
            "type": str(gemini.get("category") or "moment"),
            "timestamp": clip_end if payoff_exists else None,
            "strength": payoff,
            "reason": reason,
        },
        "risks": {
            "confusing_start": context < 50,
            "missing_context": needs_context,
            "weak_payoff": payoff < 55,
            "clickbait_gap": expectation < 50,
            "clickbait_gap_score": max(0, seed["hook"] - payoff),
            "abrupt_ending": False,
            "dead_air": False,
            "generic_outro": False,
        },
        "score_reasons": {
            "payoff_reason": reason,
            "expectation_match_reason": "Hook and payoff alignment from local reasoning/final-judge evidence.",
            "context_independence_reason": "Standalone evidence inherited from the Final Multimodal Judge.",
            "clarity_reason": "Local transcript context plus upstream standalone score.",
            "emotional_completeness_reason": "Local reasoning with upstream emotion/payoff evidence.",
            "value_density_reason": "Weighted hook/payoff/action/surprise evidence.",
            "ending_quality_reason": "Current selected ending preserves the known payoff by default.",
        },
        "protected_ranges": [],
        "ending_candidates": [
            {
                "end": value,
                "payoff_score": payoff,
                "emotional_completeness_score": emotional,
                "ending_quality_score": ending_quality if abs(value - clip_end) <= 0.05 else max(0, ending_quality - 4),
                "viewer_satisfaction_score": viewer if abs(value - clip_end) <= 0.05 else max(0, viewer - 3),
                "reason": "Conservative local ending candidate; current payoff boundary preferred.",
            }
            for value in allowed_ends
        ],
        "recommended_end": clip_end,
        "recommended_changes": [],
    }


def _normalize_editorial_angle(raw: Any, clip: dict) -> dict:
    value = raw if isinstance(raw, dict) else {}
    gemini = clip.get("gemini") if isinstance(clip.get("gemini"), dict) else {}
    return {
        "primary_angle": _clean(value.get("primary_angle") or gemini.get("category") or "moment", 120),
        "secondary_angle": _clean(value.get("secondary_angle"), 120),
        "viewer_question": _clean(value.get("viewer_question"), 200),
        "stakes": _clean(value.get("stakes") or clip.get("hook"), 200),
        "payoff": _clean(value.get("payoff") or clip.get("payoff"), 240),
        "reason": _clean(value.get("reason") or gemini.get("reason"), 240),
        "confidence": max(0.0, min(1.0, float(value.get("confidence", 0.68) or 0.68))),
    }


def _normalize_originality(raw: Any, clip: dict) -> dict:
    value = raw if isinstance(raw, dict) else {}
    seed = _multimodal_seed_scores(clip)
    context = _clamp_score(value.get("context_independence", seed["standalone"]))
    opening = _clamp_score(value.get("current_opening_quality", seed["hook"]))
    transformations = [
        _clean(item, 120)
        for item in (value.get("recommended_transformations") or [])
        if _clean(item, 120)
    ][:10]
    no_transform = bool(value.get("no_transformation_needed", False))
    if not transformations and opening >= 82 and context >= 78:
        no_transform = True
    return {
        "why_interesting": _clean(value.get("why_interesting") or (clip.get("gemini") or {}).get("reason"), 320),
        "source_dependency": _clamp_score(value.get("source_dependency", 55)),
        "context_independence": context,
        "current_opening_quality": opening,
        "transformation_need": _clamp_score(value.get("transformation_need", 55)),
        "recommended_transformations": transformations,
        "no_transformation_needed": no_transform,
    }


def _default_timeline(clip: dict) -> list[dict]:
    segments = clip.get("segments") if isinstance(clip.get("segments"), list) else []
    if segments:
        result = []
        for index, item in enumerate(segments[:8]):
            try:
                start = float(item.get("start"))
                end = float(item.get("end"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            result.append(
                {
                    "source_start": start,
                    "source_end": end,
                    "purpose": str(item.get("role") or ("payoff" if index == len(segments) - 1 else "context")),
                    "preserve_audio": True,
                    "playback_rate": 1.0,
                    "semantic_note": "Preserve selected upstream segment.",
                }
            )
        if result:
            return result
    return [
        {
            "source_start": float(clip.get("start", 0.0)),
            "source_end": float(clip.get("end", 0.0)),
            "purpose": "context",
            "preserve_audio": True,
            "playback_rate": 1.0,
            "semantic_note": "Conservative local timeline preserves selected highlight.",
        }
    ]


def _normalize_timeline(raw: Any, clip: dict, context_start: float, context_end: float) -> list[dict]:
    items = raw if isinstance(raw, list) else []
    result = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        try:
            start = max(context_start, float(item.get("source_start")))
            end = min(context_end, float(item.get("source_end")))
        except (TypeError, ValueError):
            continue
        if end <= start + 0.15:
            continue
        result.append(
            {
                "source_start": round(start, 3),
                "source_end": round(end, 3),
                "purpose": str(item.get("purpose") or "context").lower(),
                "preserve_audio": True,
                "playback_rate": 1.0,
                "semantic_note": _clean(item.get("semantic_note"), 220),
            }
        )
    return result or _default_timeline(clip)


def _normalize_proposal(raw: dict, clip: dict, context_start: float, context_end: float) -> dict:
    value = dict(raw or {})
    value["editorial_angle"] = _normalize_editorial_angle(value.get("editorial_angle"), clip)
    value["originality_analysis"] = _normalize_originality(value.get("originality_analysis"), clip)
    value["timeline"] = _normalize_timeline(value.get("timeline"), clip, context_start, context_end)

    for key in (
        "hook_candidates",
        "context_overlays",
        "visual_events",
        "audio_events",
        "caption_emphasis",
        "recommended_transformations",
    ):
        items = value.get(key)
        value[key] = list(items)[: _LIST_LIMITS[key]] if isinstance(items, list) else []

    satisfaction = value.get("viewer_satisfaction")
    if not isinstance(satisfaction, dict):
        satisfaction = _default_satisfaction(clip, context_end)
    else:
        defaults = _default_satisfaction(clip, context_end)
        merged = dict(defaults)
        merged.update(satisfaction)
        for nested in ("structure", "payoff", "risks", "score_reasons"):
            base_nested = defaults.get(nested, {})
            incoming = satisfaction.get(nested)
            merged[nested] = {**base_nested, **incoming} if isinstance(incoming, dict) else base_nested
        for key in ("protected_ranges", "ending_candidates", "recommended_changes"):
            if not isinstance(merged.get(key), list):
                merged[key] = defaults[key]
        satisfaction = merged
    value["viewer_satisfaction"] = satisfaction
    value["_provider"] = "qwen_local"
    value["_satisfaction_provider"] = "qwen_local_same_request"
    return _validate_response_shape(value)


def _validate_response_shape(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("V3 Qwen JSON invalid")
    for key in EDITORIAL_SCHEMA["required"]:
        if key not in raw:
            raise ValueError(f"V3 Qwen missing required field: {key}")
    if not isinstance(raw.get("editorial_angle"), dict):
        raise ValueError("V3 Qwen editorial_angle must be object")
    if not isinstance(raw.get("originality_analysis"), dict):
        raise ValueError("V3 Qwen originality_analysis must be object")
    for key, limit in _LIST_LIMITS.items():
        if not isinstance(raw.get(key), list):
            raise ValueError(f"V3 Qwen field must be list: {key}")
        raw[key] = raw[key][:limit]
    satisfaction = raw.get("viewer_satisfaction")
    if satisfaction is not None and not isinstance(satisfaction, dict):
        raise ValueError("V3 Qwen viewer_satisfaction must be object when present")
    return raw


def _cache_key(video_path: Path, clip: dict, profile: str, retry_feedback: list[str]) -> str:
    try:
        stat = video_path.stat()
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
    except OSError:
        size = 0
        mtime_ns = 0
    gemini = clip.get("gemini") if isinstance(clip.get("gemini"), dict) else {}
    payload = {
        "video": str(video_path.resolve()).lower(),
        "size": size,
        "mtime_ns": mtime_ns,
        "start": round(float(clip["start"]), 3),
        "end": round(float(clip["end"]), 3),
        "hook_score": clip.get("hook_score"),
        "final_judge_score": gemini.get("total_score"),
        "model": OLLAMA_MODEL,
        "profile": profile,
        "prompt_version": V3_EDITORIAL_PROMPT_VERSION,
        "provider_version": _LOCAL_PROVIDER_VERSION,
        "retry_feedback": sorted(retry_feedback),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = _LOCAL_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _save_cache(key: str, value: dict) -> None:
    _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_LOCAL_CACHE_DIR / f"{key}.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_editorial_prompt(
    *,
    clip: dict,
    transcript_text: str,
    profile: str,
    context_start: float,
    context_end: float,
    retry_feedback: list[str] | None = None,
) -> str:
    feedback = ", ".join(retry_feedback or []) or "none"
    end_candidates = build_end_candidate_values(float(clip["end"]), context_end)
    final_judge = clip.get("gemini") if isinstance(clip.get("gemini"), dict) else {}
    return f"""
You are the LOCAL QWEN editorial reasoning layer for AI Shorts V3.
Gemini has ALREADY acted as the Final Multimodal Judge upstream. Do NOT call or
imitate another multimodal judge. Use transcript evidence plus the stored final
judge metadata as your evidence. Be conservative about visuals not explicitly
supported by that metadata.

CONTENT PROFILE: {profile}
SOURCE RANGE: {float(clip['start']):.3f}s -> {float(clip['end']):.3f}s
AVAILABLE CONTEXT: {context_start:.3f}s -> {context_end:.3f}s
FINAL MULTIMODAL JUDGE EVIDENCE: {json.dumps(final_judge, ensure_ascii=False, separators=(',', ':'))}
EXISTING METADATA: {json.dumps(clip, ensure_ascii=False, separators=(',', ':'))}

CORE CONTRACT:
- Find why viewers should care and choose one truthful editorial angle.
- No invented facts, names, numbers, objects, outcomes, relationships or quotes.
- Preserve semantic truth and causal meaning.
- no_transformation_needed=true is valid when the selected moment is already strong.
- Prefer 3-6 meaningful segments, not micro-cut spam.
- timeline source_start/source_end are ABSOLUTE timestamps from the original source.
- context_overlays/visual_events/audio_events use time from the RESTRUCTURED SHORT.
- playback_rate MUST be 1.0.
- RECONSTRUCTED hooks may only use a real source quote with exact source timestamps.
- EDITORIAL hooks must be short, truthful and non-clickbait.

VIEWER SATISFACTION:
Produce viewer_satisfaction in THIS SAME QWEN REQUEST; there is no separate Gemini
Satisfaction request. Evaluate payoff, expectation match, context independence,
clarity, emotional completeness, value density and ending quality. Stored Final
Multimodal Judge scores may support visual/audio payoff evidence. Do not invent
certainty beyond them.
Allowed absolute END candidates: {json.dumps(end_candidates)}

Return JSON with:
editorial_angle, originality_analysis, hook_candidates, timeline,
context_overlays, visual_events, audio_events, caption_emphasis,
recommended_transformations, viewer_satisfaction.
Keep effects restrained and editorially justified.

RETRY WEAKNESSES FROM QA: {feedback}

TIMESTAMPED TRANSCRIPT:
{transcript_text}
""".strip()


class GeminiEditorialReasoner:
    """Compatibility name for the local Qwen V3 reasoner."""

    provider = "qwen_local"
    satisfaction_provider = "qwen_local_same_request"

    def __init__(self):
        info(
            "[V3 LOCAL] Editorial + Satisfaction provider=Qwen/Ollama "
            "(Gemini reserved for Final Multimodal Judge)."
        )

    def analyze(
        self,
        *,
        video_path: Path,
        transcript: list[dict],
        clip: dict,
        content_profile: str,
        video_duration: float,
        clip_index: int,
        retry_feedback: list[str] | None = None,
    ) -> tuple[dict, bool]:
        retry_feedback = list(retry_feedback or [])
        context_start = max(0.0, float(clip["start"]) - float(V3_CONTEXT_BEFORE))
        context_end = min(float(video_duration), float(clip["end"]) + float(V3_CONTEXT_AFTER))
        transcript_text = build_transcript_context(transcript, context_start, context_end)
        key = _cache_key(video_path, clip, content_profile, retry_feedback)
        cached = _load_cache(key)
        if cached is not None:
            info(f"[V3 LOCAL] clip={clip_index} Qwen cache hit")
            return _normalize_proposal(cached, clip, context_start, context_end), True

        prompt = build_editorial_prompt(
            clip=clip,
            transcript_text=transcript_text,
            profile=content_profile,
            context_start=context_start,
            context_end=context_end,
            retry_feedback=retry_feedback,
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
                                "Return strict JSON only. You are a conservative short-form "
                                "editor. Never invent facts; use supplied transcript and stored "
                                "Final Multimodal Judge evidence."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    format="json",
                    options={
                        "temperature": 0.12 if attempt == 1 else 0.0,
                        "num_ctx": 8192,
                        "num_predict": 3200,
                    },
                )
                raw = _parse_json(response["message"]["content"])
                proposal = _normalize_proposal(raw, clip, context_start, context_end)
                _save_cache(key, proposal)
                info(
                    f"[V3 LOCAL] clip={clip_index} Qwen editorial+satisfaction "
                    f"attempt={attempt} provider=qwen_local"
                )
                return proposal, False
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"V3 local Qwen editorial reasoning failed after retry: {last_error}")
