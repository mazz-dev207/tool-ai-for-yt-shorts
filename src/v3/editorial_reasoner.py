from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from src.config import GEMINI_API_KEY, GEMINI_MAX_RETRIES, GEMINI_MODEL, TEMP_DIR
from src.v3_config import V3_CACHE_DIR, V3_CONTEXT_AFTER, V3_CONTEXT_BEFORE, V3_EDITORIAL_PROMPT_VERSION
from src.highlights.gemini_judge import (
    GeminiQuotaExhausted, GeminiUnavailable, _is_daily_quota_exhausted,
    _is_rate_limit_error, _retry_delay_seconds, build_transcript_context,
)

EDITORIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "editorial_angle": {
            "type": "object",
            "properties": {
                "primary_angle": {"type": "string"}, "secondary_angle": {"type": "string"},
                "viewer_question": {"type": "string"}, "stakes": {"type": "string"},
                "payoff": {"type": "string"}, "reason": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["primary_angle", "secondary_angle", "viewer_question", "stakes", "payoff", "reason", "confidence"],
        },
        "originality_analysis": {
            "type": "object",
            "properties": {
                "why_interesting": {"type": "string"}, "source_dependency": {"type": "integer"},
                "context_independence": {"type": "integer"}, "current_opening_quality": {"type": "integer"},
                "transformation_need": {"type": "integer"},
                "recommended_transformations": {"type": "array", "items": {"type": "string"}},
                "no_transformation_needed": {"type": "boolean"},
            },
            "required": ["why_interesting", "source_dependency", "context_independence", "current_opening_quality", "transformation_need", "recommended_transformations", "no_transformation_needed"],
        },
        "hook_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string"}, "score": {"type": "integer"}, "text": {"type": "string"},
                    "source_start": {"type": ["number", "null"]}, "source_end": {"type": ["number", "null"]},
                    "duration": {"type": "number"}, "reason": {"type": "string"}, "confidence": {"type": "number"},
                },
                "required": ["mode", "score", "text", "source_start", "source_end", "duration", "reason", "confidence"],
            },
        },
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_start": {"type": "number"}, "source_end": {"type": "number"},
                    "purpose": {"type": "string"}, "preserve_audio": {"type": "boolean"},
                    "playback_rate": {"type": "number"}, "semantic_note": {"type": "string"},
                },
                "required": ["source_start", "source_end", "purpose", "preserve_audio", "playback_rate", "semantic_note"],
            },
        },
        "context_overlays": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "start": {"type": "number"}, "duration": {"type": "number"}, "purpose": {"type": "string"}},
                "required": ["text", "start", "duration", "purpose"],
            },
        },
        "visual_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"time": {"type": "number"}, "event": {"type": "string"}, "intensity": {"type": "number"}, "duration": {"type": "number"}, "target": {"type": "string"}, "edit": {"type": "object"}},
                "required": ["time", "event", "intensity", "duration", "target", "edit"],
            },
        },
        "audio_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"time": {"type": "number"}, "effect": {"type": "string"}, "intensity": {"type": "number"}, "duration": {"type": "number"}, "gain_db": {"type": "number"}, "asset": {"type": "string"}},
                "required": ["time", "effect", "intensity", "duration", "gain_db", "asset"],
            },
        },
        "caption_emphasis": {"type": "array", "items": {"type": "object", "properties": {"phrase": {"type": "string"}, "emphasis": {"type": "string"}}, "required": ["phrase", "emphasis"]}},
        "recommended_transformations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["editorial_angle", "originality_analysis", "hook_candidates", "timeline", "context_overlays", "visual_events", "audio_events", "caption_emphasis", "recommended_transformations"],
}


def _run(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed")


def _extract_context(video_path: Path, start: float, end: float, duration: float, clip_index: int) -> tuple[Path, float, float]:
    context_start = max(0.0, start - V3_CONTEXT_BEFORE)
    context_end = min(duration, end + V3_CONTEXT_AFTER)
    output_dir = TEMP_DIR / "v3_editorial_context"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"clip_{clip_index:03d}.mp4"
    _run(["ffmpeg", "-y", "-ss", f"{context_start:.3f}", "-to", f"{context_end:.3f}", "-i", str(video_path), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "29", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(output)])
    return output, context_start, context_end


def _cache_key(video_path: Path, clip: dict, profile: str, retry_feedback: list[str]) -> str:
    stat = video_path.stat()
    payload = {
        "video": str(video_path.resolve()).lower(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "start": round(float(clip["start"]), 3), "end": round(float(clip["end"]), 3),
        "hook_score": clip.get("hook_score"), "model": GEMINI_MODEL, "profile": profile,
        "prompt_version": V3_EDITORIAL_PROMPT_VERSION, "retry_feedback": sorted(retry_feedback),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    (V3_CACHE_DIR / f"{key}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_editorial_prompt(*, clip: dict, transcript_text: str, profile: str, context_start: float, context_end: float, retry_feedback: list[str] | None = None) -> str:
    feedback = ", ".join(retry_feedback or []) or "none"
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
- EDITORIAL: short on-screen context text. No invented facts, numbers, stakes, quotes, relationships or outcomes.
  Prefer neutral wording when uncertain. Never use generic clickbait such as YOU WON'T BELIEVE THIS.

STORY: Target Cold Open -> Context -> Escalation -> Payoff -> Reaction/Aftermath.
You MAY reorder source segments only when semantic truth and causal meaning remain intact. If event order is essential,
preserve it. A duplicated source range is allowed only for a cold-open preview, replay or callback. Keep fragmentation low.

EFFECTS: Effects are event-driven, not timer-driven. intentional > hyperactive.
For subtle events use 0-1 effects; medium 1-2; major payoff max 2-3. Fast gameplay should receive fewer disruptive effects.
Dialogue content should remain visually restrained. Sound accents must never cover dialogue; recommend conservative gain.

ORIGINALITY: Estimate source dependency and whether meaningful editorial transformation is actually useful.
It is valid to return no_transformation_needed=true if the source moment is already a strong standalone short.

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
        started = time.time(); current = uploaded
        while time.time() - started < timeout:
            state = getattr(getattr(current, "state", None), "name", None)
            if state in {None, "ACTIVE"}:
                return current
            if state == "FAILED":
                raise RuntimeError("Gemini File API processing failed")
            time.sleep(2)
            current = self.client.files.get(name=current.name)
        raise TimeoutError("Gemini File API timeout")

    def analyze(self, *, video_path: Path, transcript: list[dict], clip: dict, content_profile: str, video_duration: float, clip_index: int, retry_feedback: list[str] | None = None) -> tuple[dict, bool]:
        retry_feedback = list(retry_feedback or [])
        key = _cache_key(video_path, clip, content_profile, retry_feedback)
        cached = _load_cache(key)
        if cached is not None:
            return cached, True
        context_video, context_start, context_end = _extract_context(video_path, float(clip["start"]), float(clip["end"]), video_duration, clip_index)
        transcript_text = build_transcript_context(transcript, context_start, context_end)
        prompt = build_editorial_prompt(clip=clip, transcript_text=transcript_text, profile=content_profile, context_start=context_start, context_end=context_end, retry_feedback=retry_feedback)
        last_error = None
        for attempt in range(1, max(1, GEMINI_MAX_RETRIES) + 1):
            uploaded = None; delay = None
            try:
                uploaded = self.client.files.upload(file=str(context_video))
                uploaded = self._wait(uploaded)
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL, contents=[uploaded, prompt],
                    config=self._types.GenerateContentConfig(temperature=0.15, response_mime_type="application/json", response_json_schema=EDITORIAL_SCHEMA),
                )
                raw = json.loads(str(response.text or "").strip())
                if not isinstance(raw, dict):
                    raise ValueError("V3 Gemini JSON invalid")
                _save_cache(key, raw)
                return raw, False
            except Exception as exc:
                last_error = exc
                if _is_daily_quota_exhausted(exc):
                    raise GeminiQuotaExhausted("Gemini daily quota exhausted for V3 editorial reasoning") from exc
                if attempt < max(1, GEMINI_MAX_RETRIES):
                    fallback = min(2 ** attempt, 6)
                    delay = _retry_delay_seconds(exc, fallback) if _is_rate_limit_error(exc) else fallback
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
