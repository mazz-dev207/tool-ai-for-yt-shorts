from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama

from src.config import (
    BASE_DIR,
    HIGHLIGHTS_DIR,
    HOOK_SEARCH_AFTER,
    HOOK_SEARCH_BEFORE,
    HOOK_SEARCH_STEP,
    TRANSCRIPT_DIR,
)
from src.highlights.gemini_judge import build_transcript_context
from src.hooks.analyzer import build_hook_prompt
from src.v3.editorial_reasoner import build_editorial_prompt
from src.v3_config import V3_CONTEXT_AFTER, V3_CONTEXT_BEFORE


DEFAULT_MODELS = ["qwen3:8b", "qwen3.5:9b"]


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        return str(message.get("content") or "")
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _response_metric(response: Any, name: str) -> int | float | None:
    if isinstance(response, dict):
        value = response.get(name)
    else:
        value = getattr(response, name, None)
    if isinstance(value, (int, float)):
        return value
    return None


def _tokens_per_second(response: Any) -> float | None:
    count = _response_metric(response, "eval_count")
    duration = _response_metric(response, "eval_duration")
    if not count or not duration:
        return None
    try:
        return round(float(count) / (float(duration) / 1_000_000_000.0), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _ollama_ps() -> str:
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"ollama ps unavailable: {exc}"


def _installed_models() -> set[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except Exception as exc:
        raise RuntimeError(f"Nu pot rula 'ollama list': {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or "ollama list failed")
    names: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.add(parts[0])
    return names


def _candidate_starts(original_start: float) -> list[float]:
    step = max(0.1, float(HOOK_SEARCH_STEP))
    first = original_start - float(HOOK_SEARCH_BEFORE)
    last = original_start + float(HOOK_SEARCH_AFTER)
    values: list[float] = []
    current = first
    while current <= last + 0.001:
        value = round(max(0.0, current), 3)
        if value not in values:
            values.append(value)
        current += step
    if round(original_start, 3) not in values:
        values.append(round(original_start, 3))
        values.sort()
    return values


def _warmup(model: str) -> dict:
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": 'Return exactly this JSON object: {"ready": true}',
            }
        ],
        stream=False,
        think=False,
        keep_alive="10m",
        format="json",
        options={"temperature": 0.0, "num_ctx": 2048, "num_predict": 64},
    )
    elapsed = time.perf_counter() - started
    content = _response_content(response)
    valid = False
    try:
        valid = json.loads(content).get("ready") is True
    except Exception:
        pass
    return {
        "seconds": round(elapsed, 3),
        "valid_json": valid,
        "tokens_per_second": _tokens_per_second(response),
        "load_duration_ns": _response_metric(response, "load_duration"),
        "total_duration_ns": _response_metric(response, "total_duration"),
    }


def _run_json_prompt(
    *,
    model: str,
    prompt: str,
    num_predict: int,
) -> tuple[dict | None, dict]:
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        think=False,
        keep_alive="10m",
        format="json",
        options={
            "temperature": 0.10,
            "num_ctx": 4096,
            "num_predict": num_predict,
        },
    )
    elapsed = time.perf_counter() - started
    text = _response_content(response).strip()
    parsed = None
    error = None
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            parsed = value
        else:
            error = f"JSON root is {type(value).__name__}, expected object"
    except Exception as exc:
        error = str(exc)

    metrics = {
        "seconds": round(elapsed, 3),
        "valid_json": parsed is not None,
        "json_error": error,
        "tokens_per_second": _tokens_per_second(response),
        "prompt_eval_count": _response_metric(response, "prompt_eval_count"),
        "eval_count": _response_metric(response, "eval_count"),
        "load_duration_ns": _response_metric(response, "load_duration"),
        "total_duration_ns": _response_metric(response, "total_duration"),
        "raw_text": text,
    }
    return parsed, metrics


def _hook_shape(raw: dict | None, candidate_starts: list[float]) -> dict:
    if not isinstance(raw, dict):
        return {"valid_shape": False, "reason": "not_object"}
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        return {"valid_shape": False, "reason": "missing_candidates"}
    starts = []
    confidences = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            starts.append(round(float(item.get("start")), 3))
        except Exception:
            pass
        try:
            confidences.append(float(item.get("hook_confidence")))
        except Exception:
            pass
    expected = {round(value, 3) for value in candidate_starts}
    returned = set(starts)
    try:
        best_start = round(float(raw.get("best_start")), 3)
    except Exception:
        best_start = None
    return {
        "valid_shape": bool(candidates) and best_start is not None,
        "candidate_coverage": round(len(returned & expected) / max(1, len(expected)), 3),
        "returned_candidates": len(candidates),
        "best_start": best_start,
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
    }


def _v3_shape(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {"valid_shape": False, "reason": "not_object"}
    required = {
        "editorial_angle",
        "originality_analysis",
        "hook_candidates",
        "timeline",
        "context_overlays",
        "visual_events",
        "audio_events",
        "caption_emphasis",
        "recommended_transformations",
    }
    missing = sorted(required - set(raw))
    satisfaction = raw.get("viewer_satisfaction")
    sat_scores = {}
    if isinstance(satisfaction, dict):
        for key in (
            "viewer_satisfaction_score",
            "payoff_score",
            "expectation_match_score",
            "context_independence_score",
            "clarity_score",
            "emotional_completeness_score",
            "value_density_score",
            "ending_quality_score",
        ):
            sat_scores[key] = satisfaction.get(key)
    angle = raw.get("editorial_angle") if isinstance(raw.get("editorial_angle"), dict) else {}
    return {
        "valid_shape": not missing,
        "missing_fields": missing,
        "timeline_segments": len(raw.get("timeline") or []) if isinstance(raw.get("timeline"), list) else None,
        "hook_candidates": len(raw.get("hook_candidates") or []) if isinstance(raw.get("hook_candidates"), list) else None,
        "primary_angle": angle.get("primary_angle"),
        "viewer_question": angle.get("viewer_question"),
        "satisfaction_present": isinstance(satisfaction, dict),
        "satisfaction_scores": sat_scores,
    }


def _clip_context(transcript: list[dict], clip: dict) -> tuple[float, float, str]:
    start = float(clip.get("start", 0.0))
    end = float(clip.get("end", start))
    context_start = max(0.0, start - float(V3_CONTEXT_BEFORE))
    context_end = end + float(V3_CONTEXT_AFTER)
    return (
        context_start,
        context_end,
        build_transcript_context(transcript, context_start, context_end),
    )


def benchmark_model(
    *,
    model: str,
    clips: list[dict],
    transcript: list[dict],
    profile: str,
) -> dict:
    print(f"\n=== {model} ===")
    print("Warm-up...")
    warmup = _warmup(model)
    print(
        f"warm-up {warmup['seconds']}s | "
        f"{warmup.get('tokens_per_second')} tok/s | json={warmup['valid_json']}"
    )

    rows = []
    for index, clip in enumerate(clips, start=1):
        start = float(clip.get("start", 0.0))
        end = float(clip.get("end", start))
        starts = _candidate_starts(start)
        hook_context_start = max(0.0, starts[0] - 0.5)
        hook_context_end = end + 0.5
        hook_text = build_transcript_context(
            transcript,
            hook_context_start,
            hook_context_end,
        )
        hook_prompt = build_hook_prompt(
            original_start=start,
            highlight_end=end,
            candidate_starts=starts,
            transcript_text=hook_text,
            profile=profile,
            context_start=hook_context_start,
            context_end=hook_context_end,
        )

        print(f"[{index}/{len(clips)}] Hook START...", end=" ", flush=True)
        hook_raw, hook_metrics = _run_json_prompt(
            model=model,
            prompt=hook_prompt,
            num_predict=1800,
        )
        hook_shape = _hook_shape(hook_raw, starts)
        print(
            f"{hook_metrics['seconds']}s | {hook_metrics.get('tokens_per_second')} tok/s | "
            f"json={hook_metrics['valid_json']} shape={hook_shape.get('valid_shape')}"
        )

        context_start, context_end, context_text = _clip_context(transcript, clip)
        v3_prompt = build_editorial_prompt(
            clip=clip,
            transcript_text=context_text,
            profile=profile,
            context_start=context_start,
            context_end=context_end,
            retry_feedback=[],
        )
        print(f"[{index}/{len(clips)}] V3 + Satisfaction...", end=" ", flush=True)
        v3_raw, v3_metrics = _run_json_prompt(
            model=model,
            prompt=v3_prompt,
            num_predict=3600,
        )
        v3_shape = _v3_shape(v3_raw)
        print(
            f"{v3_metrics['seconds']}s | {v3_metrics.get('tokens_per_second')} tok/s | "
            f"json={v3_metrics['valid_json']} shape={v3_shape.get('valid_shape')}"
        )

        rows.append(
            {
                "clip_index": index,
                "source_start": start,
                "source_end": end,
                "hook": {
                    "metrics": hook_metrics,
                    "shape": hook_shape,
                    "output": hook_raw,
                },
                "v3_satisfaction": {
                    "metrics": v3_metrics,
                    "shape": v3_shape,
                    "output": v3_raw,
                },
            }
        )

    def avg(path: tuple[str, ...]) -> float | None:
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return round(sum(values) / len(values), 3) if values else None

    return {
        "model": model,
        "warmup": warmup,
        "summary": {
            "avg_hook_seconds": avg(("hook", "metrics", "seconds")),
            "avg_hook_tokens_per_second": avg(("hook", "metrics", "tokens_per_second")),
            "avg_v3_seconds": avg(("v3_satisfaction", "metrics", "seconds")),
            "avg_v3_tokens_per_second": avg(("v3_satisfaction", "metrics", "tokens_per_second")),
            "hook_json_success_rate": round(
                sum(1 for row in rows if row["hook"]["metrics"]["valid_json"]) / max(1, len(rows)),
                3,
            ),
            "hook_shape_success_rate": round(
                sum(1 for row in rows if row["hook"]["shape"].get("valid_shape")) / max(1, len(rows)),
                3,
            ),
            "v3_json_success_rate": round(
                sum(1 for row in rows if row["v3_satisfaction"]["metrics"]["valid_json"]) / max(1, len(rows)),
                3,
            ),
            "v3_shape_success_rate": round(
                sum(1 for row in rows if row["v3_satisfaction"]["shape"].get("valid_shape")) / max(1, len(rows)),
                3,
            ),
        },
        "clips": rows,
        "ollama_ps_after_model": _ollama_ps(),
    }


def _print_summary(results: list[dict]) -> None:
    print("\n=== SUMMARY ===")
    header = (
        f"{'MODEL':<18} {'HOOK s':>9} {'HOOK tok/s':>11} {'HOOK JSON':>10} "
        f"{'V3 s':>9} {'V3 tok/s':>10} {'V3 JSON':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        s = result["summary"]
        print(
            f"{result['model']:<18} "
            f"{str(s['avg_hook_seconds']):>9} "
            f"{str(s['avg_hook_tokens_per_second']):>11} "
            f"{str(s['hook_shape_success_rate']):>10} "
            f"{str(s['avg_v3_seconds']):>9} "
            f"{str(s['avg_v3_tokens_per_second']):>10} "
            f"{str(s['v3_shape_success_rate']):>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local Ollama models on the real AI Shorts Hook START and "
            "V3+Viewer Satisfaction prompts without changing production config."
        )
    )
    parser.add_argument("video_name", help="e.g. video1")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama model tags to compare",
    )
    parser.add_argument("--clips", type=int, default=3, help="number of real highlights to test")
    parser.add_argument("--profile", default="gaming", help="content profile")
    args = parser.parse_args()

    installed = _installed_models()
    missing = [model for model in args.models if model not in installed]
    if missing:
        print("Lipsesc modele Ollama:")
        for model in missing:
            print(f"  ollama pull {model}")
        return 2

    transcript_path = TRANSCRIPT_DIR / f"{args.video_name}.json"
    highlights_path = HIGHLIGHTS_DIR / f"{args.video_name}.json"
    transcript = _load_json(transcript_path)
    highlights = _load_json(highlights_path)
    if not isinstance(transcript, list):
        raise ValueError(f"Transcript invalid: {transcript_path}")
    if not isinstance(highlights, list) or not highlights:
        raise ValueError(f"Highlights invalid/empty: {highlights_path}")

    clips = highlights[: max(1, min(len(highlights), args.clips))]
    print("Qwen A/B benchmark - production config remains unchanged")
    print(f"Video: {args.video_name}")
    print(f"Clips: {len(clips)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Profile: {args.profile}")

    results = []
    for model in args.models:
        results.append(
            benchmark_model(
                model=model,
                clips=clips,
                transcript=transcript,
                profile=args.profile,
            )
        )

    _print_summary(results)

    output_dir = BASE_DIR / "output" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{args.video_name}_qwen_ab_{stamp}.json"
    payload = {
        "benchmark_version": "qwen-ab-v1",
        "video": args.video_name,
        "profile": args.profile,
        "tested_clips": len(clips),
        "models": args.models,
        "production_config_changed": False,
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nRaport complet: {output_path}")
    print("Trimite-mi acel JSON si comparam inclusiv calitatea editoriala, nu doar viteza.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
