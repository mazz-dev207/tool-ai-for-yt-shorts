from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.config import GEMINI_API_KEY, GEMINI_MODEL, HIGHLIGHTS_DIR, OUTPUT_DIR
from src.highlights.gemini_judge import (
    GeminiQuotaExhausted,
    GeminiUnavailable,
    _is_daily_quota_exhausted,
    _is_rate_limit_error,
    _retry_delay_seconds,
)
from src.logger import info, warning
from src.v3_config import (
    V3_SATISFACTION_CACHE_DIR,
    V3_SATISFACTION_FINAL_VALIDATION,
    V3_SATISFACTION_MAX_REFINEMENT_PASSES,
    V3_SATISFACTION_MAX_RETRIES,
)


FINAL_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "viewer_satisfaction_score": {"type": "integer"},
        "hook_score": {"type": "integer"},
        "context_score": {"type": "integer"},
        "clarity_score": {"type": "integer"},
        "pacing_score": {"type": "integer"},
        "payoff_score": {"type": "integer"},
        "ending_quality_score": {"type": "integer"},
        "originality_score": {"type": "integer"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "refinement": {
            "type": "object",
            "properties": {
                "needed": {"type": "boolean"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "start_shift_seconds": {"type": "number"},
                "end_shift_seconds": {"type": "number"},
            },
            "required": [
                "needed", "reasons", "start_shift_seconds", "end_shift_seconds"
            ],
        },
    },
    "required": [
        "viewer_satisfaction_score", "hook_score", "context_score",
        "clarity_score", "pacing_score", "payoff_score",
        "ending_quality_score", "originality_score", "problems", "strengths",
        "refinement",
    ],
}


def _cache_key(path: Path, profile: str, clip_metadata: dict) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()).lower(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "profile": str(profile or "auto"),
        "metadata": {
            "hook": (clip_metadata.get("v3") or {}).get("hook_score_v3"),
            "satisfaction": (clip_metadata.get("v3") or {}).get("viewer_satisfaction_score"),
            "final_score": (clip_metadata.get("v3") or {}).get("final_score"),
        },
        "version": "final-satisfaction-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = V3_SATISFACTION_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _save_cache(key: str, value: dict) -> None:
    V3_SATISFACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (V3_SATISFACTION_CACHE_DIR / f"{key}.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _validate_payload(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Final Satisfaction JSON invalid")
    for key in FINAL_VALIDATION_SCHEMA["required"]:
        if key not in raw:
            raise ValueError(f"Final Satisfaction missing field: {key}")
    if not isinstance(raw.get("refinement"), dict):
        raise ValueError("Final Satisfaction refinement must be object")
    raw["problems"] = list(raw.get("problems", []) or [])[:8]
    raw["strengths"] = list(raw.get("strengths", []) or [])[:8]
    raw["refinement"]["reasons"] = list(
        raw["refinement"].get("reasons", []) or []
    )[:6]
    return raw


class GeminiFinalSatisfactionValidator:
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
                raise RuntimeError("Gemini final validation file processing failed")
            time.sleep(2.0)
            current = self.client.files.get(name=current.name)
        raise TimeoutError("Gemini final satisfaction validation timeout")

    def analyze(
        self,
        *,
        rendered_path: Path,
        content_profile: str,
        clip_metadata: dict,
        clip_index: int,
    ) -> tuple[dict, bool]:
        key = _cache_key(rendered_path, content_profile, clip_metadata)
        cached = _load_cache(key)
        if cached is not None:
            return _validate_payload(cached), True

        prompt = f"""
You are the optional FINAL Viewer Satisfaction validator for AI Shorts V3.
Review the fully rendered Short multimodally: video, dialogue/audio, pacing, subtitles,
visual framing, hook, context, payoff and ending.

CONTENT PROFILE: {content_profile}
CLIP INDEX: {clip_index}
PRE-RENDER METADATA: {json.dumps(clip_metadata, ensure_ascii=False, separators=(',', ':'))}

Judge whether watching the finished Short feels worth it. Do not optimize only for scroll-stop.
Check Hook, Context, Clarity, Pacing, Payoff, Ending, Originality and overall Satisfaction.
A visual payoff or facial/gameplay reaction counts even without dialogue.
A pause is not dead air when it creates tension, comedy, anticipation or emotional impact.

If a repair is clearly justified, refinement.needed=true and recommend only a conservative
start/end shift. Do not request a rewrite for subjective style preferences. The caller allows at
most {max(0, int(V3_SATISFACTION_MAX_REFINEMENT_PASSES))} refinement pass(es); this validator
only requests refinement and never loops by itself.

Return JSON only.
""".strip()

        last_error = None
        max_attempts = max(1, int(V3_SATISFACTION_MAX_RETRIES))
        for attempt in range(1, max_attempts + 1):
            uploaded = None
            delay = None
            try:
                uploaded = self.client.files.upload(file=str(rendered_path))
                uploaded = self._wait(uploaded)
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[uploaded, prompt],
                    config=self._types.GenerateContentConfig(
                        temperature=0.10,
                        response_mime_type="application/json",
                        response_json_schema=FINAL_VALIDATION_SCHEMA,
                    ),
                )
                raw = _validate_payload(
                    json.loads(str(response.text or "").strip())
                )
                _save_cache(key, raw)
                return raw, False
            except Exception as exc:
                last_error = exc
                if _is_daily_quota_exhausted(exc):
                    raise GeminiQuotaExhausted(
                        "Gemini daily quota exhausted for final Satisfaction validation"
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
        raise RuntimeError(f"Final Satisfaction validation failed: {last_error}")


def validate_final_outputs(
    video_name: str,
    rendered_paths: list[Path],
    content_profile: str,
) -> list[dict]:
    if not V3_SATISFACTION_FINAL_VALIDATION:
        return []

    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    try:
        clips = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warning(f"[SATISFACTION][FINAL] highlights unavailable: {exc}")
        return []
    if not isinstance(clips, list):
        return []

    try:
        validator = GeminiFinalSatisfactionValidator()
    except Exception as exc:
        warning(f"[SATISFACTION][FINAL] validator unavailable: {exc}")
        return []

    results: list[dict] = []
    quota_exhausted = False
    for index, rendered_path in enumerate(rendered_paths, start=1):
        clip = clips[index - 1] if index - 1 < len(clips) else {}
        if quota_exhausted:
            result = {
                "status": "unavailable",
                "reason": "gemini_daily_quota_exhausted",
            }
        else:
            try:
                raw, from_cache = validator.analyze(
                    rendered_path=Path(rendered_path),
                    content_profile=content_profile,
                    clip_metadata=clip,
                    clip_index=index,
                )
                result = {
                    "status": "available",
                    "from_cache": from_cache,
                    **raw,
                    "refinement_status": (
                        "requested_not_applied"
                        if bool((raw.get("refinement") or {}).get("needed", False))
                        and V3_SATISFACTION_MAX_REFINEMENT_PASSES > 0
                        else "not_requested"
                    ),
                    "max_refinement_passes": V3_SATISFACTION_MAX_REFINEMENT_PASSES,
                }
            except GeminiQuotaExhausted as exc:
                quota_exhausted = True
                warning(f"[SATISFACTION][FINAL] {exc}")
                result = {"status": "unavailable", "reason": str(exc)}
            except Exception as exc:
                warning(
                    f"[SATISFACTION][FINAL] clip={index} validation failed: {exc}"
                )
                result = {"status": "unavailable", "reason": str(exc)}

        results.append(result)
        if index - 1 < len(clips):
            clips[index - 1]["final_satisfaction_validation"] = result
        debug = OUTPUT_DIR / "debug" / video_name / f"clip_{index}"
        debug.mkdir(parents=True, exist_ok=True)
        (debug / "final_satisfaction_validation.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if result.get("status") == "available":
            info(
                f"[SATISFACTION][FINAL] clip={index} "
                f"score={result.get('viewer_satisfaction_score')} "
                f"payoff={result.get('payoff_score')} "
                f"ending={result.get('ending_quality_score')}"
            )
            if result.get("refinement_status") == "requested_not_applied":
                info(
                    f"[SATISFACTION][FINAL] clip={index} refinement requested: "
                    f"{(result.get('refinement') or {}).get('reasons', [])}"
                )

    highlights_path.write_text(
        json.dumps(clips, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results
