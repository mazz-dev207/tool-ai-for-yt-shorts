from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from src.config import GEMINI_API_KEY, GEMINI_MAX_RETRIES, GEMINI_MODEL, TEMP_DIR
from src.satisfaction.satisfaction_analyzer import build_end_candidate_values
from src.v3_config import (
    V3_CACHE_DIR,
    V3_CONTEXT_AFTER,
    V3_CONTEXT_BEFORE,
    V3_EDITORIAL_PROMPT_VERSION,
)
from src.highlights.gemini_judge import (
    GeminiQuotaExhausted,
    GeminiUnavailable,
    _is_daily_quota_exhausted,
    _is_rate_limit_error,
    _retry_delay_seconds,
    build_transcript_context,
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
        "hook_promise": {"type": "string"},
        "actual_payoff": {"type": "string"},
        "structure": {
            "type": "object",
            "properties": {
                "setup": {"type": "boolean"},
                "tension": {"type": "boolean"},
                "escalation": {"type": "boolean"},
                "payoff": {"type": "boolean"},
                "pattern": {"type": "string"},
            },
            "required": ["setup", "tension", "escalation", "payoff", "pattern"],
        },
        "payoff": {
            "type": "object",
            "properties": {
                "exists": {"type": "boolean"},
                "type": {"type": "string"},
                "timestamp": {"type": ["number", "null"]},
                "strength": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["exists", "type", "timestamp", "strength", "reason"],
        },
        "risks": {
            "type": "object",
            "properties": {
                "confusing_start": {"type": "boolean"},
                "missing_context": {"type": "boolean"},
                "weak_payoff": {"type": "boolean"},
                "clickbait_gap": {"type": "boolean"},
                "clickbait_gap_score": {"type": "integer"},
                "abrupt_ending": {"type": "boolean"},
                "dead_air": {"type": "boolean"},
                "generic_outro": {"type": "boolean"},
            },
            "required": [
                "confusing_start", "missing_context", "weak_payoff",
                "clickbait_gap", "clickbait_gap_score", "abrupt_ending",
                "dead_air", "generic_outro",
            ],
        },
        "score_reasons": {
            "type": "object",
            "properties": {
                "payoff_reason": {"type": "string"},
                "expectation_match_reason": {"type": "string"},
                "context_independence_reason": {"type": "string"},
                "clarity_reason": {"type": "string"},
                "emotional_completeness_reason": {"type": "string"},
                "value_density_reason": {"type": "string"},
                "ending_quality_reason": {"type": "string"},
            },
            "required": [
                "payoff_reason", "expectation_match_reason",
                "context_independence_reason", "clarity_reason",
                "emotional_completeness_reason", "value_density_reason",
                "ending_quality_reason",
            ],
        },
        "protected_ranges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "reason"],
            },
        },
        "ending_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "end": {"type": "number"},
                    "payoff_score": {"type": "integer"},
                    "emotional_completeness_score": {"type": "integer"},
                    "ending_quality_score": {"type": "integer"},
                    "viewer_satisfaction_score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "end", "payoff_score", "emotional_completeness_score",
                    "ending_quality_score", "viewer_satisfaction_score", "reason",
                ],
            },
        },
        "recommended_end": {"type": ["number", "null"]},
        "recommended_changes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "viewer_satisfaction_score", "payoff_score", "expectation_match_score",
        "context_independence_score", "clarity_score",
        "emotional_completeness_score", "value_density_score",
        "ending_quality_score", "hook_promise", "actual_payoff", "structure",
        "payoff", "risks", "score_reasons", "protected_ranges",
        "ending_candidates", "recommended_end", "recommended_changes",
    ],
}


EDITORIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "editorial_angle": {
            "type": "object",
            "properties": {
                "primary_angle": {"type": "string"},
                "secondary_angle": {"type": "string"},
                "viewer_question": {"type": "string"},
                "stakes": {"type": "string"},
                "payoff": {"type": "string"},
                "reason": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "primary_angle", "secondary_angle", "viewer_question",
                "stakes", "payoff", "reason", "confidence",
            ],
        },
        "originality_analysis": {
            "type": "object",
            "properties": {
                "why_interesting": {"type": "string"},
                "source_dependency": {"type": "integer"},
                "context_independence": {"type": "integer"},
                "current_opening_quality": {"type": "integer"},
                "transformation_need": {"type": "integer"},
                "recommended_transformations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "no_transformation_needed": {"type": "boolean"},
            },
            "required": [
                "why_interesting", "source_dependency", "context_independence",
                "current_opening_quality", "transformation_need",
                "recommended_transformations", "no_transformation_needed",
            ],
        },
        "hook_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string"},
                    "score": {"type": "integer"},
                    "text": {"type": "string"},
                    "source_start": {"type": ["number", "null"]},
                    "source_end": {"type": ["number", "null"]},
                    "duration": {"type": "number"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "mode", "score", "text", "source_start", "source_end",
                    "duration", "reason", "confidence",
                ],
            },
        },
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_start": {"type": "number"},
                    "source_end": {"type": "number"},
                    "purpose": {"type": "string"},
                    "preserve_audio": {"type": "boolean"},
                    "playback_rate": {"type": "number"},
                    "semantic_note": {"type": "string"},
                },
                "required": [
                    "source_start", "source_end", "purpose", "preserve_audio",
                    "playback_rate", "semantic_note",
                ],
            },
        },
        "context_overlays": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "start": {"type": "number"},
                    "duration": {"type": "number"},
                    "purpose": {"type": "string"},
                },
                "required": ["text", "start", "duration", "purpose"],
            },
        },
        "visual_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "number"},
                    "event": {"type": "string"},
                    "intensity": {"type": "number"},
                    "duration": {"type": "number"},
                    "target": {"type": "string"},
                    "edit": {"type": "object"},
                },
                "required": [
                    "time", "event", "intensity", "duration", "target", "edit",
                ],
            },
        },
        "audio_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "number"},
                    "effect": {"type": "string"},
                    "intensity": {"type": "number"},
                    "duration": {"type": "number"},
                    "gain_db": {"type": "number"},
                    "asset": {"type": "string"},
                },
                "required": [
                    "time", "effect", "intensity", "duration", "gain_db", "asset",
                ],
            },
        },
        "caption_emphasis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string"},
                    "emphasis": {"type": "string"},
                },
                "required": ["phrase", "emphasis"],
            },
        },
        "recommended_transformations": {
            "type": "array",
            "items": {"type": "string"},
        },
        # Optional at the outer schema level so a malformed/missing Satisfaction
        # block never invalidates an otherwise usable editorial plan. The engine
        # will mark Satisfaction unavailable and preserve existing ranking.
        "viewer_satisfaction": VIEWER_SATISFACTION_SCHEMA,
    },
    "required": [
        "editorial_angle", "originality_analysis", "hook_candidates", "timeline",
        "context_overlays", "visual_events", "audio_events", "caption_emphasis",
        "recommended_transformations",
    ],
}


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed")


def _extract_context(
    video_path: Path,
    start: float,
    end: float,
    duration: float,
    clip_index: int,
) -> tuple[Path, float, float]:
    context_start = max(0.0, start - V3_CONTEXT_BEFORE)
    context_end = min(duration, end + V3_CONTEXT_AFTER)
    output_dir = TEMP_DIR / "v3_editorial_context"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"clip_{clip_index:03d}.mp4"
    _run(
        [
            "ffmpeg", "-y",
            "-ss", f"{context_start:.3f}",
            "-to", f"{context_end:.3f}",
            "-i", str(video_path),
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "29",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(output),
        ]
    )
    return output, context_start, context_end


def _cache_key(
    video_path: Path,
    clip: dict,
    profile: str,
    retry_feedback: list[str],
) -> str:
    stat = video_path.stat()
    payload = {
        "video": str(video_path.resolve()).lower(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "start": round(float(clip["start"]), 3),
        "end": round(float(clip["end"]), 3),
        "hook_score": clip.get("hook_score"),
        "model": GEMINI_MODEL,
        "profile": profile,
        "prompt_version": V3_EDITORIAL_PROMPT_VERSION,
        "retry_feedback": sorted(retry_feedback),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = V3_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _save_cache(key: str, value: dict) -> None:
    V3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (V3_CACHE_DIR / f"{key}.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _validate_response_shape(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("V3 Gemini JSON invalid")
    for key in EDITORIAL_SCHEMA["required"]:
        if key not in raw:
            raise ValueError(f"V3 Gemini missing required field: {key}")
    for key in (
        "hook_candidates", "timeline", "context_overlays", "visual_events",
        "audio_events", "caption_emphasis", "recommended_transformations",
    ):
        if not isinstance(raw.get(key), list):
            raise ValueError(f"V3 Gemini field must be list: {key}")

    raw["hook_candidates"] = raw["hook_candidates"][:6]
    raw["timeline"] = raw["timeline"][:8]
    raw["context_overlays"] = raw["context_overlays"][:4]
    raw["visual_events"] = raw["visual_events"][:10]
    raw["audio_events"] = raw["audio_events"][:6]
    raw["caption_emphasis"] = raw["caption_emphasis"][:8]
    raw["recommended_transformations"] = raw["recommended_transformations"][:10]

    satisfaction = raw.get("viewer_satisfaction")
    if satisfaction is not None and not isinstance(satisfaction, dict):
        raw.pop("viewer_satisfaction", None)
    elif isinstance(satisfaction, dict):
        for key, limit in (
            ("protected_ranges", 10),
            ("ending_candidates", 6),
            ("recommended_changes", 8),
        ):
            value = satisfaction.get(key)
            if isinstance(value, list):
                satisfaction[key] = value[:limit]
            elif value is not None:
                satisfaction[key] = []
    return raw


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
    return f"""
You are the editorial reasoning layer for AI Shorts V3.
The highlight and Hook Optimizer analysis already exist. Do NOT replace highlight ranking.
Turn this selected moment into a truthful, standalone short-form story.

CONTENT PROFILE: {profile}
SOURCE RANGE: {float(clip['start']):.3f}s -> {float(clip['end']):.3f}s
AVAILABLE CONTEXT: {context_start:.3f}s -> {context_end:.3f}s
EXISTING METADATA: {json.dumps(clip, ensure_ascii=False, separators=(',', ':'))}

CORE PRINCIPLE: Don't just extract the moment. Find why viewers should care, choose an editorial angle,
build the strongest truthful opening, compress setup, preserve causality and payoff, then propose only semantic edits
that add comprehension, stakes, curiosity or clarity.

HOOK MODES:
- NATIVE: existing source opening.
- RECONSTRUCTED: a REAL later source quote/reaction can become a cold open. It must use exact real source audio,
  source_start/source_end must identify it, and it must not change speaker meaning.
- EDITORIAL: short on-screen context text. No invented facts, numbers, stakes, quotes, relationships, outcomes or creator intentions.
  Prefer neutral wording when uncertain. Never use generic clickbait such as YOU WON'T BELIEVE THIS.

STORY:
- Target Cold Open -> Context -> Escalation -> Payoff -> Reaction/Aftermath.
- timeline.source_start/source_end are ABSOLUTE timestamps from the original source video.
- Every source range MUST remain inside AVAILABLE CONTEXT.
- playback_rate MUST be 1.0 in this V3 version.
- You MAY reorder source segments only when semantic truth and causal meaning remain intact.
- If event order is essential, preserve it.
- A duplicated source range is allowed only for a cold-open preview, replay or callback.
- Keep fragmentation low; prefer 3-6 meaningful segments over many micro-cuts.

OUTPUT-TIMELINE TIME BASE:
- context_overlays.start is seconds from the beginning of the RESTRUCTURED SHORT.
- visual_events.time is seconds from the beginning of the RESTRUCTURED SHORT.
- audio_events.time is seconds from the beginning of the RESTRUCTURED SHORT.
- These are NOT absolute source timestamps.

EFFECTS:
- Effects are event-driven, not timer-driven. intentional > hyperactive.
- Subtle event: 0-1 effects. Medium: max 1-2. Major payoff: max 2-3 coordinated effects.
- Fast gameplay should receive fewer disruptive effects.
- Dialogue content should remain visually restrained.
- Sound accents must never cover dialogue; recommend conservative gain.
- Prefer supported semantic operations such as punch-in/focus emphasis over decorative effects.

ORIGINALITY:
Estimate source dependency and whether meaningful editorial transformation is actually useful.
It is valid to return no_transformation_needed=true if the source moment is already a strong standalone short.
Originality means meaningful editorial value, not arbitrary effects and not platform-detection evasion.

VIEWER SATISFACTION — REQUIRED WHEN EVIDENCE IS SUFFICIENT:
Evaluate the COMPLETE viewer experience, not just scroll-stop or watch time. Use the uploaded VIDEO + AUDIO together
with transcript/timestamps and EXISTING METADATA. A visual gameplay death, facial reaction, reveal, silence, camera change
or physical result can be payoff even when transcript text does not say it.

Score independently (0-100): payoff quality, hook-to-payoff expectation match, context independence, clarity,
emotional completeness, value density, and ending quality. Keep each reason evidence-based and short.
Do not reward exaggerated hooks. Explicitly describe HOOK PROMISE and ACTUAL PAYOFF and flag clickbait_gap when delivery
is materially weaker/different than the promise.

Content-aware mini-structures:
- gaming: action -> problem/stakes -> reaction/escalation -> payoff/result
- podcast/interview: claim -> curiosity -> explanation -> insight/conclusion
- entertainment: setup -> expectation/challenge -> escalation/twist -> reaction/result
- reaction: trigger -> anticipation -> reaction -> interpretation/payoff
- story: curiosity/setup -> escalation -> resolution
These are guides, not mandatory rigid templates.

VALUE DENSITY:
Dead air is low-value silence/filler. Do NOT call a pause dead air when it creates tension, comedy, anticipation,
emotional weight or makes a reaction readable. Put such source-grounded moments in protected_ranges using ABSOLUTE
source timestamps and a reason such as comedic_pause, tension, anticipation, reaction, visual_payoff or necessary_context.

ENDING OPTIMIZATION:
Evaluate ONLY these allowed absolute END candidates: {json.dumps(end_candidates)}
For each useful candidate, score payoff, emotional completeness, ending quality and overall satisfaction.
recommended_end MUST be one of those values or null. Choose the ending that best completes the viewer experience,
not the shortest ending. Penalize generic outro, dead air after payoff, and abrupt ending before reaction/resolution.

The viewer_satisfaction_score you return is advisory/evidence only; deterministic code recalculates the final Satisfaction
score from sub-scores and later combines it with Hook, Retention and Originality using configured weights.
If evidence is genuinely insufficient, keep reasons explicit rather than inventing certainty.

RETRY WEAKNESSES FROM QA: {feedback}

TIMESTAMPED TRANSCRIPT:
{transcript_text}

Return JSON only matching the schema. Keep explanations short.
""".strip()


class GeminiEditorialReasoner:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise GeminiUnavailable("GEMINI_API_KEY lipsește")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiUnavailable("Pachetul google-genai lipsește") from exc
        self._types = types
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def _wait(self, uploaded: Any, timeout: float = 180.0) -> Any:
        started = time.time()
        current = uploaded
        while time.time() - started < timeout:
            state = getattr(getattr(current, "state", None), "name", None)
            if state in {None, "ACTIVE"}:
                return current
            if state == "FAILED":
                raise RuntimeError("Gemini File API processing failed")
            time.sleep(2.0)
            current = self.client.files.get(name=current.name)
        raise TimeoutError("Gemini File API timeout")

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
        key = _cache_key(
            video_path,
            clip,
            content_profile,
            retry_feedback,
        )
        cached = _load_cache(key)
        if cached is not None:
            return _validate_response_shape(cached), True

        context_video, context_start, context_end = _extract_context(
            video_path,
            float(clip["start"]),
            float(clip["end"]),
            video_duration,
            clip_index,
        )
        transcript_text = build_transcript_context(
            transcript,
            context_start,
            context_end,
        )
        prompt = build_editorial_prompt(
            clip=clip,
            transcript_text=transcript_text,
            profile=content_profile,
            context_start=context_start,
            context_end=context_end,
            retry_feedback=retry_feedback,
        )

        last_error = None
        max_attempts = max(1, GEMINI_MAX_RETRIES)
        for attempt in range(1, max_attempts + 1):
            uploaded = None
            delay = None
            try:
                uploaded = self.client.files.upload(file=str(context_video))
                uploaded = self._wait(uploaded)
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[uploaded, prompt],
                    config=self._types.GenerateContentConfig(
                        temperature=0.15,
                        response_mime_type="application/json",
                        response_json_schema=EDITORIAL_SCHEMA,
                    ),
                )
                raw = json.loads(str(response.text or "").strip())
                raw = _validate_response_shape(raw)
                _save_cache(key, raw)
                return raw, False

            except Exception as exc:
                last_error = exc
                if _is_daily_quota_exhausted(exc):
                    raise GeminiQuotaExhausted(
                        "Gemini daily quota exhausted for V3 editorial reasoning"
                    ) from exc
                if attempt < max_attempts:
                    fallback = min(2 ** attempt, 6)
                    delay = (
                        _retry_delay_seconds(exc, fallback)
                        if _is_rate_limit_error(exc)
                        else fallback
                    )

            finally:
                if uploaded is not None:
                    try:
                        self.client.files.delete(name=uploaded.name)
                    except Exception:
                        pass

            if delay is not None:
                time.sleep(delay)
                continue
            break

        raise RuntimeError(f"V3 editorial reasoning failed: {last_error}")
