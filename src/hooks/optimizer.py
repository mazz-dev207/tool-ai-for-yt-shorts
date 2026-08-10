from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.config import (
    GEMINI_MAX_REFINED_DURATION,
    GEMINI_MIN_REFINED_DURATION,
    HIGHLIGHTS_DIR,
    HOOK_MIN_CONFIDENCE,
    HOOK_OPTIMIZER_ENABLED,
    TRANSCRIPT_DIR,
)
from src.highlights.gemini_judge import GeminiQuotaExhausted, probe_duration
from src.hooks.analyzer import GeminiHookAnalyzer
from src.hooks.candidates import generate_hook_start_candidates
from src.hooks.models import HookOptimizationResult
from src.hooks.scoring import (
    clamp_semantic_start,
    normalize_hook_candidate,
    refine_start_locally,
    select_best_hook_candidate,
)
from src.logger import info, success, warning


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _highlight_score(clip: dict) -> int:
    sources = [
        (clip.get("gemini") or {}).get("total_score"),
        clip.get("score"),
        (clip.get("scores") or {}).get("viral_potential"),
        clip.get("retention_score"),
    ]
    for value in sources:
        try:
            if value is not None:
                return max(0, min(100, int(round(float(value)))))
        except (TypeError, ValueError):
            continue
    return 0


def _duration_for_clip(clip: dict) -> float:
    segments = clip.get("segments") or []
    if segments:
        total = 0.0
        for segment in segments:
            try:
                total += max(0.0, float(segment["end"]) - float(segment["start"]))
            except (KeyError, TypeError, ValueError):
                continue
        return total
    return max(0.0, float(clip["end"]) - float(clip["start"]))


def _visual_boundary_score(video_path: Path, start: float) -> float:
    """Tiny tie-break signal for a clean shot/action boundary; never decisive alone."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0
    frames = []
    try:
        for time_value in (max(0.0, start - 0.10), start + 0.06):
            capture.set(cv2.CAP_PROP_POS_MSEC, time_value * 1000.0)
            ok, frame = capture.read()
            if not ok:
                return 0.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            width = gray.shape[1]
            if width > 480:
                scale = 480.0 / width
                gray = cv2.resize(
                    gray,
                    (480, max(2, int(gray.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(gray)
    finally:
        capture.release()
    if len(frames) != 2 or frames[0].shape != frames[1].shape:
        return 0.0
    difference = float(cv2.absdiff(frames[0], frames[1]).mean()) / 255.0
    # Kept intentionally small: language/context continuity dominates a near tie.
    return round(min(0.8, difference * 4.0), 4)


def _fallback_result(clip: dict, reason: str) -> HookOptimizationResult:
    start = float(clip["start"])
    return HookOptimizationResult(
        original_start=start,
        original_end=float(clip["end"]),
        semantic_start=start,
        optimized_start=start,
        applied=False,
        fallback_reason=reason,
    )


def _apply_optimized_start(clip: dict, optimized_start: float) -> tuple[bool, str | None]:
    optimized_start = float(optimized_start)
    segments = [dict(item) for item in (clip.get("segments") or [])]

    if segments:
        try:
            first_index = min(range(len(segments)), key=lambda index: float(segments[index]["start"]))
            first = segments[first_index]
            if optimized_start >= float(first["end"]) - 0.35:
                return False, "optimized_start_would_destroy_first_segment"
            first["start"] = optimized_start
            duration = sum(
                max(0.0, float(item["end"]) - float(item["start"]))
                for item in segments
            )
            if not (GEMINI_MIN_REFINED_DURATION <= duration <= GEMINI_MAX_REFINED_DURATION):
                return False, "optimized_start_breaks_duration_limits"
            clip["segments"] = segments
            clip["start"] = round(min(float(item["start"]) for item in segments), 3)
            clip["end"] = round(max(float(item["end"]) for item in segments), 3)
            clip["duration"] = round(duration, 3)
            return True, None
        except (KeyError, TypeError, ValueError):
            return False, "invalid_extract_segments"

    end = float(clip["end"])
    duration = end - optimized_start
    if not (GEMINI_MIN_REFINED_DURATION <= duration <= GEMINI_MAX_REFINED_DURATION):
        return False, "optimized_start_breaks_duration_limits"
    clip["start"] = round(optimized_start, 3)
    clip["duration"] = round(duration, 3)
    return True, None


def optimize_hook_start(
    *,
    video_path: Path,
    transcript: list[dict],
    highlight: dict,
    content_profile: str,
    clip_index: int = 1,
    analyzer=None,
    video_duration: float | None = None,
) -> HookOptimizationResult:
    original_start = float(highlight["start"])
    original_end = float(highlight["end"])

    if not HOOK_OPTIMIZER_ENABLED:
        return _fallback_result(highlight, "hook_optimizer_disabled")

    if video_duration is None:
        video_duration = probe_duration(video_path)

    candidate_starts = generate_hook_start_candidates(
        original_start,
        original_end,
        video_duration,
    )
    if not candidate_starts:
        return _fallback_result(highlight, "no_start_candidates")

    analyzer = analyzer or GeminiHookAnalyzer()
    raw, from_cache = analyzer.analyze(
        video_path=video_path,
        transcript=transcript,
        original_start=original_start,
        highlight_end=original_end,
        candidate_starts=candidate_starts,
        profile=content_profile,
        video_duration=video_duration,
        clip_index=clip_index,
    )

    normalized = []
    lower = original_start - 1e-3 - max(0.0, original_start - min(candidate_starts))
    upper = max(candidate_starts) + 1e-3
    allowed = set(round(value, 3) for value in candidate_starts)
    for item in raw.get("candidates", []):
        try:
            candidate = normalize_hook_candidate(item)
        except Exception:
            continue
        # Gemini may round slightly, but it may not invent arbitrary candidate starts.
        nearest = min(candidate_starts, key=lambda value: abs(value - candidate.start))
        if abs(nearest - candidate.start) > 0.12:
            continue
        candidate.start = round(nearest, 3)
        if candidate.start not in allowed or not (lower <= candidate.start <= upper):
            continue
        candidate.continuity_score += _visual_boundary_score(video_path, candidate.start)
        normalized.append(candidate)

    best = select_best_hook_candidate(normalized, transcript, original_start)
    if best is None:
        return _fallback_result(highlight, "no_valid_gemini_hook_candidates")

    if best.confidence < HOOK_MIN_CONFIDENCE:
        result = _fallback_result(highlight, "hook_confidence_below_threshold")
        result.hook_score = best.hook_score
        result.hook_type = best.hook_type
        result.secondary_hook_type = best.secondary_hook_type
        result.confidence = best.confidence
        result.hook_reason = best.reason
        result.hook_scores = best.scores
        result.hook_penalties = best.penalties
        result.loop_score = best.loop_score
        result.candidates = [item.to_dict() for item in normalized]
        result.from_cache = from_cache
        return result

    semantic_start = clamp_semantic_start(best.start, original_start, original_end)
    optimized_start = refine_start_locally(
        semantic_start,
        transcript,
        original_start,
        original_end,
    )

    result = HookOptimizationResult(
        original_start=original_start,
        original_end=original_end,
        semantic_start=semantic_start,
        optimized_start=optimized_start,
        hook_score=best.hook_score,
        hook_type=best.hook_type,
        secondary_hook_type=best.secondary_hook_type,
        confidence=best.confidence,
        hook_reason=best.reason,
        hook_scores=best.scores,
        hook_penalties=best.penalties,
        loop_score=best.loop_score,
        candidates=[item.to_dict() for item in normalized],
        from_cache=from_cache,
    )

    applied, reason = _apply_optimized_start(highlight, optimized_start)
    result.applied = applied
    if not applied:
        result.optimized_start = original_start
        result.fallback_reason = reason
    return result


def _attach_metadata(clip: dict, result: HookOptimizationResult) -> None:
    clip["highlight_score"] = _highlight_score(clip)
    clip["hook_score"] = result.hook_score
    clip["hook_type"] = result.hook_type
    clip["hook_confidence"] = round(result.confidence, 4)
    clip["original_start"] = round(result.original_start, 3)
    clip["optimized_start"] = round(result.optimized_start, 3)
    clip["hook_shift"] = result.hook_shift
    clip["loop_score"] = result.loop_score
    clip["hook_optimizer"] = result.to_dict()


def optimize_hooks(
    video_name: str,
    video_path: Path,
    content_profile: str = "auto",
) -> Path:
    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"
    clips = _load_json(highlights_path)
    transcript = _load_json(transcript_path)
    if not isinstance(clips, list) or not clips:
        warning("[HOOK] Nu există highlights pentru Hook Optimizer.")
        return highlights_path
    if not isinstance(transcript, list):
        warning("[HOOK] Transcript invalid; păstrez start-urile originale.")
        return highlights_path

    if not HOOK_OPTIMIZER_ENABLED:
        info("[HOOK] Hook Optimizer dezactivat; pipeline-ul rămâne neschimbat.")
        return highlights_path

    video_duration = probe_duration(video_path)
    try:
        analyzer = GeminiHookAnalyzer()
    except Exception as exc:
        warning(f"[HOOK] Gemini indisponibil: {exc}; păstrez toate start-urile originale.")
        for clip in clips:
            _attach_metadata(clip, _fallback_result(clip, f"gemini_unavailable: {exc}"))
        _save_json(highlights_path, clips)
        return highlights_path

    debug_dir = HIGHLIGHTS_DIR / "debug" / video_name / "hooks"
    debug_dir.mkdir(parents=True, exist_ok=True)
    metadata_items = []
    quota_exhausted = False

    info(f"[HOOK] Optimizez start-ul pentru {len(clips)} Shorts...")
    for index, clip in enumerate(clips, start=1):
        original_snapshot = json.loads(json.dumps(clip))
        if quota_exhausted:
            result = _fallback_result(clip, "gemini_daily_quota_exhausted")
        else:
            try:
                result = optimize_hook_start(
                    video_path=video_path,
                    transcript=transcript,
                    highlight=clip,
                    content_profile=content_profile,
                    clip_index=index,
                    analyzer=analyzer,
                    video_duration=video_duration,
                )
            except GeminiQuotaExhausted as exc:
                quota_exhausted = True
                warning(f"[HOOK {index}/{len(clips)}] {exc}; opresc request-urile Hook Gemini.")
                result = _fallback_result(clip, "gemini_daily_quota_exhausted")
            except Exception as exc:
                warning(f"[HOOK {index}/{len(clips)}] eșuat: {exc}; folosesc original_start.")
                result = _fallback_result(clip, f"hook_error: {exc}")

        _attach_metadata(clip, result)
        debug_payload = {
            "before": original_snapshot,
            "result": result.to_dict(),
            "after": clip,
        }
        _save_json(debug_dir / f"clip_{index}.json", debug_payload)
        metadata_items.append({
            "clip_index": index,
            "highlight_score": clip.get("highlight_score", 0),
            "hook_score": result.hook_score,
            "original_start": result.original_start,
            "semantic_start": result.semantic_start,
            "optimized_start": result.optimized_start,
            "hook_shift": result.hook_shift,
            "hook_type": result.hook_type,
            "secondary_hook_type": result.secondary_hook_type,
            "hook_confidence": result.confidence,
            "loop_score": result.loop_score,
            "applied": result.applied,
            "fallback_reason": result.fallback_reason,
            "performance": {
                "viewed_vs_swiped": None,
                "average_view_duration": None,
                "audience_retention": None,
            },
        })

        status = "APPLIED" if result.applied else "FALLBACK"
        info(
            f"[HOOK {index}/{len(clips)}] {status} | "
            f"{result.original_start:.2f}s -> {result.optimized_start:.2f}s | "
            f"score={result.hook_score} confidence={result.confidence:.2f} "
            f"type={result.hook_type}"
        )

    _save_json(highlights_path, clips)
    _save_json(
        HIGHLIGHTS_DIR / f"{video_name}_hook_metadata.json",
        {
            "video": video_path.name,
            "content_profile": content_profile,
            "quota_exhausted": quota_exhausted,
            "shorts": metadata_items,
        },
    )
    success(f"[HOOK] Hook optimization terminat pentru {len(clips)} Shorts.")
    success(f"[HOOK] Debug JSON: {debug_dir}")
    return highlights_path
