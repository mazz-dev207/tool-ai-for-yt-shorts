from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    HOOK_CACHE_DIR,
    HOOK_OPTIMIZER_VERSION,
    HOOK_SEARCH_AFTER,
    HOOK_SEARCH_BEFORE,
    TEMP_DIR,
)
from src.highlights.gemini_judge import (
    GeminiQuotaExhausted,
    GeminiUnavailable,
    _is_daily_quota_exhausted,
    _is_rate_limit_error,
    _retry_delay_seconds,
    build_transcript_context,
)
from src.hooks.scoring import profile_priority_text


HOOK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "hook_score": {"type": "integer"},
                    "hook_type": {"type": "string"},
                    "secondary_hook_type": {"type": ["string", "null"]},
                    "hook_scores": {
                        "type": "object",
                        "properties": {
                            "immediate_action": {"type": "integer"},
                            "curiosity_gap": {"type": "integer"},
                            "emotional_reaction": {"type": "integer"},
                            "conflict_tension": {"type": "integer"},
                            "visual_surprise": {"type": "integer"},
                            "context_independence": {"type": "integer"},
                            "payoff_proximity": {"type": "integer"},
                        },
                        "required": [
                            "immediate_action", "curiosity_gap", "emotional_reaction",
                            "conflict_tension", "visual_surprise", "context_independence",
                            "payoff_proximity",
                        ],
                    },
                    "hook_penalties": {
                        "type": "object",
                        "properties": {
                            "dead_air": {"type": "integer"},
                            "context_dependency": {"type": "integer"},
                            "spoiled_payoff": {"type": "integer"},
                            "mid_sentence": {"type": "integer"},
                            "duplicate_information": {"type": "integer"},
                        },
                        "required": [
                            "dead_air", "context_dependency", "spoiled_payoff",
                            "mid_sentence", "duplicate_information",
                        ],
                    },
                    "hook_reason": {"type": "string"},
                    "confidence": {"type": "number"},
                    "loop_score": {"type": "integer"},
                },
                "required": [
                    "start", "hook_score", "hook_type", "secondary_hook_type",
                    "hook_scores", "hook_penalties", "hook_reason", "confidence", "loop_score",
                ],
            },
        },
        "best_start": {"type": "number"},
    },
    "required": ["candidates", "best_start"],
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


def _extract_hook_context(
    video_path: Path,
    original_start: float,
    highlight_end: float,
    video_duration: float,
    clip_index: int,
) -> tuple[Path, float, float]:
    start = max(0.0, original_start - HOOK_SEARCH_BEFORE - 0.50)
    end = min(video_duration, highlight_end + 0.50)
    if end <= start:
        raise ValueError("Invalid Hook Optimizer context range")

    output_dir = TEMP_DIR / "hook_context"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"hook_{clip_index:03d}.mp4"
    _run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(video_path),
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "29",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(output),
    ])
    return output, start, end


def _cache_key(
    video_path: Path,
    original_start: float,
    highlight_end: float,
    profile: str,
    candidate_starts: list[float],
) -> str:
    stat = video_path.stat()
    payload = {
        "video": str(video_path.resolve()).lower(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "highlight_start": round(original_start, 3),
        "highlight_end": round(highlight_end, 3),
        "candidate_starts": [round(value, 3) for value in candidate_starts],
        "model": GEMINI_MODEL,
        "profile": profile,
        "version": HOOK_OPTIMIZER_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = HOOK_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(key: str, value: dict) -> None:
    HOOK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (HOOK_CACHE_DIR / f"{key}.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_hook_prompt(
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
You are the HOOK START OPTIMIZER for short-form video.
The highlight has ALREADY been selected. Do NOT judge whether the whole highlight is good.
Your only job is to decide where the Short should START to maximize first-second retention.
Highlight Score and Hook Score are separate concepts.

Use ALL supplied modalities together: VIDEO FRAMES + AUDIO + TIMESTAMPED TRANSCRIPT.
Do not score from keywords alone. Visual stakes, facial reaction, action, silence, speech intensity,
and whether the viewer understands the situation must all matter.

CONTENT PROFILE: {profile}
PROFILE PRIORITY: {profile_priority_text(profile)}

ORIGINAL HIGHLIGHT: {original_start:.3f}s -> {highlight_end:.3f}s
AVAILABLE MULTIMODAL CONTEXT: {context_start:.3f}s -> {context_end:.3f}s
CANDIDATE STARTS: {json.dumps(candidate_starts, separators=(',', ':'))}

For EVERY candidate start, score:
- immediate_action 0-20
- curiosity_gap 0-20
- emotional_reaction 0-15
- conflict_tension 0-15
- visual_surprise 0-10
- context_independence 0-10
- payoff_proximity 0-10

Apply NEGATIVE penalties where appropriate:
- dead_air 0 to -20
- context_dependency 0 to -20
- spoiled_payoff 0 to -30
- mid_sentence 0 to -15
- duplicate_information 0 to -20

Central rule: prefer MINIMUM CONTEXT + MAXIMUM CURIOSITY BEFORE THE PAYOFF.
Do not start directly on the kill, explosion, punchline, reveal or final result when the preceding
moment creates necessary stakes. Preserve setup -> tension -> payoff with the least setup possible.
Reaction-first is valid only when the reaction naturally occurs before the reveal/payoff; never reorder time.
Penalize starts that cut an essential sentence, start with references to missing prior context, or reveal the result.

Hook types: ACTION, REACTION, CURIOSITY, CONFLICT, SURPRISE, STAKES, QUESTION,
PREDICTION, PAYOFF_TEASE, VISUAL_WTF, MIXED.
Reason: one short debugging sentence only.
Confidence: 0.0-1.0, reflecting certainty that this exact start is better than nearby alternatives.
loop_score: 0-100 only for natural loop potential; never invent an artificial loop.

TIMESTAMPED TRANSCRIPT:
{transcript_text}

Return every supplied candidate exactly once and return best_start.
""".strip()


class GeminiHookAnalyzer:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise GeminiUnavailable("GEMINI_API_KEY lipsește")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiUnavailable(
                "Pachetul google-genai lipsește. Instalează: pip install google-genai"
            ) from exc
        self._types = types
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def _wait_until_active(self, uploaded: Any, timeout_seconds: float = 180.0) -> Any:
        started = time.time()
        current = uploaded
        while time.time() - started < timeout_seconds:
            state = getattr(getattr(current, "state", None), "name", None)
            if state in {None, "ACTIVE"}:
                return current
            if state == "FAILED":
                raise RuntimeError("Gemini File API processing failed")
            time.sleep(2.0)
            current = self.client.files.get(name=current.name)
        raise TimeoutError("Gemini File API processing timeout")

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
        key = _cache_key(
            video_path, original_start, highlight_end, profile, candidate_starts
        )
        cached = _load_cache(key)
        if cached is not None:
            return cached, True

        context_video, context_start, context_end = _extract_hook_context(
            video_path, original_start, highlight_end, video_duration, clip_index
        )
        transcript_text = build_transcript_context(transcript, context_start, context_end)
        prompt = build_hook_prompt(
            original_start=original_start,
            highlight_end=highlight_end,
            candidate_starts=candidate_starts,
            transcript_text=transcript_text,
            profile=profile,
            context_start=context_start,
            context_end=context_end,
        )

        max_attempts = max(1, GEMINI_MAX_RETRIES)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            uploaded = None
            retry_delay: float | None = None
            try:
                uploaded = self.client.files.upload(file=str(context_video))
                uploaded = self._wait_until_active(uploaded)
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[uploaded, prompt],
                    config=self._types.GenerateContentConfig(
                        temperature=0.10,
                        response_mime_type="application/json",
                        response_json_schema=HOOK_RESPONSE_SCHEMA,
                    ),
                )
                raw = json.loads(str(response.text or "").strip())
                if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), list):
                    raise ValueError("Gemini Hook Optimizer JSON invalid")
                _save_cache(key, raw)
                return raw, False
            except Exception as exc:
                last_error = exc
                if _is_daily_quota_exhausted(exc):
                    raise GeminiQuotaExhausted(
                        f"Gemini daily quota exhausted for Hook Optimizer model {GEMINI_MODEL}"
                    ) from exc
                if attempt < max_attempts:
                    fallback_delay = min(2 ** attempt, 6)
                    retry_delay = (
                        _retry_delay_seconds(exc, fallback_delay)
                        if _is_rate_limit_error(exc)
                        else float(fallback_delay)
                    )
            finally:
                if uploaded is not None:
                    try:
                        self.client.files.delete(name=uploaded.name)
                    except Exception:
                        pass

            if retry_delay is not None:
                time.sleep(retry_delay)
                continue
            break

        raise RuntimeError(f"Gemini Hook Optimizer failed after retries: {last_error}")
