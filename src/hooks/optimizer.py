from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.config import (
    GEMINI_MAX_REFINED_DURATION,
    GEMINI_MIN_REFINED_DURATION,
    HIGHLIGHTS_DIR,
    HOOK_DEBUG,
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
    """Read the existing highlight score without mixing Hook Score into it."""
    sources = [
        clip.get("highlight_score"),
        (clip.get("gemini") or {}).get("total_score"),
        clip.get("score"),
        (clip.get("scores") or {}).get("viral_potential"),
    ]
    for value in sources:
        try:
            if value is not None:
                return max(0, min(100, int(round(float(value)))))
        except (TypeError, ValueError):
            continue
    return 0


def _visual_boundary_score(video_path: Path, start: float) -> float:
    """Small shot/motion continuity tie-break signal; semantic scores remain dominant."""
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


def _duration(segments: list[dict]) -> float:
    return sum(
        max(0.0, float(item["end"]) - float(item["start"]))
        for item in segments
    )


def _apply_optimized_start(
    clip: dict,
    optimized_start: float,
    video_duration: float,
) -> tuple[bool, str | None]:
    """Apply START safely; only extend END when required to keep minimum duration."""
    optimized_start = max(0.0, min(float(optimized_start), float(video_duration)))
    segments = [dict(item) for item in (clip.get("segments") or [])]

    if segments:
        try:
            first_index = min(
                range(len(segments)),
                key=lambda index: float(segments[index]["start"]),
            )
            last_index = max(
                range(len(segments)),
                key=lambda index: float(segments[index]["end"]),
            )
            first = segments[first_index]
            if optimized_start >= float(first["end"]) - 0.35:
                return False, "optimized_start_would_destroy_first_segment"
            first["start"] = optimized_start

            duration = _duration(segments)
            if duration < GEMINI_MIN_REFINED_DURATION:
                missing = GEMINI_MIN_REFINED_DURATION - duration
                old_end = float(segments[last_index]["end"])
                new_end = min(float(video_duration), old_end + missing)
                segments[last_index]["end"] = new_end
                duration = _duration(segments)

            if duration < GEMINI_MIN_REFINED_DURATION - 1e-3:
                return False, "optimized_start_cannot_preserve_min_duration"
            if duration > GEMINI_MAX_REFINED_DURATION + 1e-3:
                return False, "optimized_start_breaks_max_duration"

            clip["segments"] = segments
            clip["start"] = round(
                min(float(item["start"]) for item in segments),
                3,
            )
            clip["end"] = round(
                max(float(item["end"]) for item in segments),
                3,
            )
            clip["duration"] = round(duration, 3)
            return True, None
        except (KeyError, TypeError, ValueError):
            return False, "invalid_extract_segments"

    try:
        end = float(clip["end"])
    except (KeyError, TypeError, ValueError):
        return False, "invalid_highlight_end"

    duration = end - optimized_start
    if duration < GEMINI_MIN_REFINED_DURATION:
        end = min(
            float(video_duration),
            optimized_start + GEMINI_MIN_REFINED_DURATION,
        )
        duration = end - optimized_start

    if duration < GEMINI_MIN_REFINED_DURATION - 1e-3:
        return False, "optimized_start_cannot_preserve_min_duration"
    if duration > GEMINI_MAX_REFINED_DURATION + 1e-3:
        return False, "optimized_start_breaks_max_duration"

    clip["start"] = round(optimized_start, 3)
    clip["end"] = round(end, 3)
    clip["duration"] = round(duration, 3)
    return True, None


def _copy_candidate_result(
    result: HookOptimizationResult,
    candidate,
    normalized: list,
    from_cache: bool,
) -> HookOptimizationResult:
    result.base_hook_score = candidate.base_hook_score
    result.hook_score = candidate.final_hook_score
    result.hook_type = candidate.hook_type
    result.secondary_hook_types = list(candidate.secondary_hook_types)
    result.confidence = candidate.confidence
    result.hook_reason = candidate.reason
    result.core_scores = dict(candidate.core_scores)
    result.human_modifiers = dict(candidate.human_modifiers)
    result.penalties = dict(candidate.penalties)
    result.loop_score = candidate.loop_score
    result.candidates = [item.to_dict() for item in normalized]
    result.from_cache = from_cache
    return result


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
    allowed = set(round(value, 3) for value in candidate_starts)
    lower = min(candidate_starts) - 1e-3
    upper = max(candidate_starts) + 1e-3
    for item in raw.get("candidates", []):
        try:
            candidate = normalize_hook_candidate(item)
        except Exception:
            continue
        nearest = min(candidate_starts, key=lambda value: abs(value - candidate.start))
        if abs(nearest - candidate.start) > 0.12:
            continue
        candidate.start = round(nearest, 3)
        if candidate.start not in allowed or not (lower <= candidate.start <= upper):
            continue
        candidate.continuity_score = _visual_boundary_score(
            video_path,
            candidate.start,
        )
        normalized.append(candidate)

    best = select_best_hook_candidate(
        normalized,
        transcript,
        original_start,
    )
    if best is None:
        return _fallback_result(highlight, "no_valid_gemini_hook_candidates")

    if best.confidence < HOOK_MIN_CONFIDENCE:
        result = _fallback_result(highlight, "hook_confidence_below_threshold")
        return _copy_candidate_result(result, best, normalized, from_cache)

    semantic_start = clamp_semantic_start(
        best.start,
        original_start,
        original_end,
    )
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
    )
    _copy_candidate_result(result, best, normalized, from_cache)

    applied, reason = _apply_optimized_start(
        highlight,
        optimized_start,
        video_duration,
    )
    result.applied = applied
    if not applied:
        result.optimized_start = original_start
        result.fallback_reason = reason
    return result


def _attach_metadata(clip: dict, result: HookOptimizationResult) -> None:
    # Keep Highlight Score and Hook Score explicitly separate.
    clip["highlight_score"] = _highlight_score(clip)
    clip["base_hook_score"] = result.base_hook_score
    clip["hook_score"] = result.hook_score
    clip["hook_type"] = result.hook_type
    clip["secondary_hook_types"] = list(result.secondary_hook_types)
    clip["hook_confidence"] = round(result.confidence, 4)
    clip["hook_original_start"] = round(result.original_start, 3)
    clip["optimized_start"] = round(result.optimized_start, 3)
    clip["hook_shift"] = result.hook_shift
    clip["contradiction_score"] = int(result.human_modifiers.get("contradiction", 0) or 0)
    clip["specificity_score"] = int(result.human_modifiers.get("specificity", 0) or 0)
    clip["timeframe_tension_score"] = int(result.human_modifiers.get("timeframe_tension", 0) or 0)
    clip["relatability_score"] = int(result.human_modifiers.get("relatability", 0) or 0)
    clip["naturalness_score"] = int(result.human_modifiers.get("naturalness", 0) or 0)
    clip["forced_hook_penalty"] = int(result.penalties.get("forced_hook", 0) or 0)
    clip["loop_score"] = result.loop_score
    clip["hook_optimizer"] = result.to_dict()


def _debug_log(index: int, total: int, result: HookOptimizationResult) -> None:
    if not HOOK_DEBUG:
        return
    info("========================================")
    info("HOOK OPTIMIZATION")
    info("========================================")
    info(
        f"Highlight: {result.original_start:.2f} -> {result.original_end:.2f}"
    )
    info("Candidates:")
    for candidate in result.candidates:
        marker = " <-- BEST" if abs(
            float(candidate.get("start", 0.0)) - result.semantic_start
        ) <= 0.01 else ""
        info(
            f"{float(candidate.get('start', 0.0)):.2f} -> "
            f"Base {int(candidate.get('base_hook_score', 0))} -> "
            f"Final {int(candidate.get('final_hook_score', 0))}{marker}"
        )
    info(f"Original start: {result.original_start:.2f}")
    info(f"Optimized semantic start: {result.semantic_start:.2f}")
    info(f"Frame-adjusted start: {result.optimized_start:.2f}")
    info(f"Shift: {result.hook_shift:+.2f}s")
    info(f"Hook type: {result.hook_type}")
    if result.secondary_hook_types:
        info(f"Secondary: {', '.join(result.secondary_hook_types)}")
    info(
        "Human modifiers: "
        f"contradiction={result.human_modifiers.get('contradiction', 0):+d} | "
        f"specificity={result.human_modifiers.get('specificity', 0):+d} | "
        f"timeframe={result.human_modifiers.get('timeframe_tension', 0):+d} | "
        f"relatability={result.human_modifiers.get('relatability', 0):+d} | "
        f"naturalness={result.human_modifiers.get('naturalness', 0):+d}"
    )
    info(
        "Penalties: "
        f"forced_hook={result.penalties.get('forced_hook', 0)} | "
        f"dead_air={result.penalties.get('dead_air', 0)} | "
        f"spoiled_payoff={result.penalties.get('spoiled_payoff', 0)} | "
        f"mid_sentence={result.penalties.get('mid_sentence', 0)}"
    )
    info(f"Confidence: {result.confidence:.2f}")
    if result.hook_reason:
        info(f"Reason: {result.hook_reason}")
    info("========================================")


def optimize_hooks(
    video_name: str,
    video_path: Path,
    content_profile: str = "auto",
) -> Path:
    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"

    # Feature flag means exact artifact compatibility: do not rewrite JSON at all.
    if not HOOK_OPTIMIZER_ENABLED:
        info("[HOOK] Hook Optimizer dezactivat; highlights rămân nemodificate.")
        return highlights_path

    clips = _load_json(highlights_path)
    transcript = _load_json(transcript_path)
    if not isinstance(clips, list) or not clips:
        warning("[HOOK] Nu există highlights pentru Hook Optimizer.")
        return highlights_path
    if not isinstance(transcript, list):
        warning("[HOOK] Transcript invalid; păstrez start-urile originale.")
        return highlights_path

    try:
        video_duration = probe_duration(video_path)
    except Exception as exc:
        warning(f"[HOOK] Nu pot determina durata video: {exc}; păstrez START-urile.")
        return highlights_path

    try:
        analyzer = GeminiHookAnalyzer()
    except Exception as exc:
        warning(
            f"[HOOK] Gemini indisponibil: {exc}; păstrez toate start-urile originale."
        )
        for clip in clips:
            _attach_metadata(
                clip,
                _fallback_result(clip, f"gemini_unavailable: {exc}"),
            )
        _save_json(highlights_path, clips)
        return highlights_path

    debug_dir = HIGHLIGHTS_DIR / "debug" / video_name / "hooks"
    debug_dir.mkdir(parents=True, exist_ok=True)
    metadata_items = []
    quota_exhausted = False

    info(f"[HOOK] Optimizez START-ul pentru {len(clips)} Shorts...")
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
                warning(
                    f"[HOOK {index}/{len(clips)}] {exc}; opresc request-urile Hook Gemini."
                )
                result = _fallback_result(clip, "gemini_daily_quota_exhausted")
            except Exception as exc:
                warning(
                    f"[HOOK {index}/{len(clips)}] eșuat: {exc}; folosesc original_start."
                )
                result = _fallback_result(clip, f"hook_error: {exc}")

        _attach_metadata(clip, result)
        _debug_log(index, len(clips), result)

        if HOOK_DEBUG:
            _save_json(
                debug_dir / f"clip_{index}.json",
                {
                    "before": original_snapshot,
                    "result": result.to_dict(),
                    "after": clip,
                },
            )

        metadata_items.append(
            {
                "clip_index": index,
                "highlight_score": clip.get("highlight_score", 0),
                "base_hook_score": result.base_hook_score,
                "hook_score": result.hook_score,
                "original_start": result.original_start,
                "semantic_start": result.semantic_start,
                "optimized_start": result.optimized_start,
                "hook_shift": result.hook_shift,
                "hook_type": result.hook_type,
                "secondary_hook_types": result.secondary_hook_types,
                "contradiction_score": result.human_modifiers.get("contradiction", 0),
                "specificity_score": result.human_modifiers.get("specificity", 0),
                "timeframe_tension_score": result.human_modifiers.get("timeframe_tension", 0),
                "relatability_score": result.human_modifiers.get("relatability", 0),
                "naturalness_score": result.human_modifiers.get("naturalness", 0),
                "forced_hook_penalty": result.penalties.get("forced_hook", 0),
                "hook_confidence": result.confidence,
                "loop_score": result.loop_score,
                "applied": result.applied,
                "fallback_reason": result.fallback_reason,
                "performance": {
                    "viewed_vs_swiped": None,
                    "average_view_duration": None,
                    "audience_retention": None,
                    "rewatch_rate": None,
                    "likes": None,
                    "comments": None,
                    "shares": None,
                },
            }
        )

        status = "APPLIED" if result.applied else "FALLBACK"
        info(
            f"[HOOK {index}/{len(clips)}] {status} | "
            f"{result.original_start:.2f}s -> {result.optimized_start:.2f}s | "
            f"base={result.base_hook_score} final={result.hook_score} "
            f"confidence={result.confidence:.2f} type={result.hook_type}"
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
    if HOOK_DEBUG:
        success(f"[HOOK] Debug JSON: {debug_dir}")
    return highlights_path
