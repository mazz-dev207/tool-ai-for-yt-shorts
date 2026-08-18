from __future__ import annotations

import argparse
import json
import subprocess
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
from src.hooks.qwen_analyzer import build_qwen_hook_prompt, _sanitize_payload
from src.v3.editorial_reasoner import build_editorial_prompt, _normalize_proposal
from src.v3_config import V3_CONTEXT_AFTER, V3_CONTEXT_BEFORE


DEFAULT_MODELS = ["qwen3:8b", "qwen3.5:9b"]

HOOK_SYSTEM = "Return strict JSON only. Score hook starts conservatively and truthfully."
V3_SYSTEM = (
    "Return strict JSON only. You are a conservative short-form editor. "
    "Never invent facts; use supplied transcript and stored Final Multimodal Judge evidence."
)


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
    return value if isinstance(value, (int, float)) else None


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
    result = subprocess.run(
        ["ollama", "list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
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
    original = round(original_start, 3)
    if original not in values:
        values.append(original)
        values.sort()
    return values


def _warmup(model: str) -> dict:
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": 'Return exactly this JSON object: {"ready": true}'}],
        stream=False,
        think=False,
        keep_alive="10m",
        format="json",
        options={"temperature": 0.0, "num_ctx": 2048, "num_predict": 64},
    )
    elapsed = time.perf_counter() - started
    text = _response_content(response).strip()
    valid = False
    try:
        valid = json.loads(text).get("ready") is True
    except Exception:
        pass
    return {
        "seconds": round(elapsed, 3),
        "valid_json": valid,
        "tokens_per_second": _tokens_per_second(response),
        "load_duration_ns": _response_metric(response, "load_duration"),
        "total_duration_ns": _response_metric(response, "total_duration"),
    }


def _run_json(*, model: str, system: str, prompt: str, num_ctx: int, num_predict: int, temperature: float) -> tuple[dict | None, dict]:
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        stream=False,
        think=False,
        keep_alive="10m",
        format="json",
        options={"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
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
    return parsed, {
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


def _hook_validation(raw: dict | None, candidate_starts: list[float]) -> tuple[dict | None, dict]:
    if not isinstance(raw, dict):
        return None, {"production_valid": False, "error": "not_object"}
    try:
        sanitized = _sanitize_payload(raw, candidate_starts)
    except Exception as exc:
        return None, {"production_valid": False, "error": str(exc)}
    returned = [float(item["start"]) for item in sanitized.get("candidates", [])]
    expected = {round(v, 3) for v in candidate_starts}
    got = {round(v, 3) for v in returned}
    return sanitized, {
        "production_valid": True,
        "candidate_coverage": round(len(expected & got) / max(1, len(expected)), 3),
        "returned_candidates": len(returned),
        "best_start": sanitized.get("best_start"),
    }


def _v3_validation(raw: dict | None, clip: dict, context_start: float, context_end: float) -> tuple[dict | None, dict]:
    if not isinstance(raw, dict):
        return None, {"production_valid": False, "error": "not_object"}
    try:
        normalized = _normalize_proposal(raw, clip, context_start, context_end)
    except Exception as exc:
        return None, {"production_valid": False, "error": str(exc)}
    sat = normalized.get("viewer_satisfaction") if isinstance(normalized.get("viewer_satisfaction"), dict) else {}
    angle = normalized.get("editorial_angle") if isinstance(normalized.get("editorial_angle"), dict) else {}
    return normalized, {
        "production_valid": True,
        "timeline_segments": len(normalized.get("timeline") or []),
        "hook_candidates": len(normalized.get("hook_candidates") or []),
        "primary_angle": angle.get("primary_angle"),
        "viewer_question": angle.get("viewer_question"),
        "viewer_satisfaction_score": sat.get("viewer_satisfaction_score"),
        "payoff_score": sat.get("payoff_score"),
        "ending_quality_score": sat.get("ending_quality_score"),
    }


def _clip_context(transcript: list[dict], clip: dict) -> tuple[float, float, str]:
    start = float(clip.get("start", 0.0))
    end = float(clip.get("end", start))
    context_start = max(0.0, start - float(V3_CONTEXT_BEFORE))
    context_end = end + float(V3_CONTEXT_AFTER)
    return context_start, context_end, build_transcript_context(transcript, context_start, context_end)


def benchmark_model(*, model: str, clips: list[dict], transcript: list[dict], profile: str) -> dict:
    print(f"\n=== {model} ===")
    print("Warm-up...")
    warmup = _warmup(model)
    print(f"warm-up {warmup['seconds']}s | {warmup.get('tokens_per_second')} tok/s | json={warmup['valid_json']}")
    rows = []
    for index, clip in enumerate(clips, start=1):
        start = float(clip.get("start", 0.0))
        end = float(clip.get("end", start))
        starts = _candidate_starts(start)
        hook_context_start = max(0.0, starts[0] - 0.5)
        hook_context_end = end + 0.5
        hook_text = build_transcript_context(transcript, hook_context_start, hook_context_end)
        hook_prompt = build_qwen_hook_prompt(
            original_start=start,
            highlight_end=end,
            candidate_starts=starts,
            transcript_text=hook_text,
            profile=profile,
            context_start=hook_context_start,
            context_end=hook_context_end,
        )
        print(f"[{index}/{len(clips)}] Hook START production prompt...", end=" ", flush=True)
        hook_raw, hook_metrics = _run_json(model=model, system=HOOK_SYSTEM, prompt=hook_prompt, num_ctx=4096, num_predict=1800, temperature=0.10)
        hook_normalized, hook_validation = _hook_validation(hook_raw, starts)
        print(f"{hook_metrics['seconds']}s | {hook_metrics.get('tokens_per_second')} tok/s | json={hook_metrics['valid_json']} production={hook_validation['production_valid']}")

        context_start, context_end, context_text = _clip_context(transcript, clip)
        v3_prompt = build_editorial_prompt(clip=clip, transcript_text=context_text, profile=profile, context_start=context_start, context_end=context_end, retry_feedback=[])
        print(f"[{index}/{len(clips)}] V3 + Satisfaction production prompt...", end=" ", flush=True)
        v3_raw, v3_metrics = _run_json(model=model, system=V3_SYSTEM, prompt=v3_prompt, num_ctx=8192, num_predict=3200, temperature=0.12)
        v3_normalized, v3_validation = _v3_validation(v3_raw, clip, context_start, context_end)
        print(f"{v3_metrics['seconds']}s | {v3_metrics.get('tokens_per_second')} tok/s | json={v3_metrics['valid_json']} production={v3_validation['production_valid']}")

        rows.append({
            "clip_index": index,
            "source_start": start,
            "source_end": end,
            "hook": {"metrics": hook_metrics, "validation": hook_validation, "raw_output": hook_raw, "production_output": hook_normalized},
            "v3_satisfaction": {"metrics": v3_metrics, "validation": v3_validation, "raw_output": v3_raw, "production_output": v3_normalized},
        })

    def avg(key: str, stage: str):
        values = [row[stage]["metrics"].get(key) for row in rows]
        nums = [float(v) for v in values if isinstance(v, (int, float))]
        return round(sum(nums) / len(nums), 3) if nums else None

    summary = {
        "avg_hook_seconds": avg("seconds", "hook"),
        "avg_hook_tokens_per_second": avg("tokens_per_second", "hook"),
        "hook_production_success_rate": round(sum(bool(row["hook"]["validation"].get("production_valid")) for row in rows) / len(rows), 3),
        "avg_v3_seconds": avg("seconds", "v3_satisfaction"),
        "avg_v3_tokens_per_second": avg("tokens_per_second", "v3_satisfaction"),
        "v3_production_success_rate": round(sum(bool(row["v3_satisfaction"]["validation"].get("production_valid")) for row in rows) / len(rows), 3),
    }
    return {"model": model, "warmup": warmup, "summary": summary, "clips": rows, "ollama_ps_after_model": _ollama_ps()}


def _print_summary(results: list[dict]) -> None:
    print("\n=== PRODUCTION-LIKE SUMMARY ===")
    print(f"{'MODEL':<18} {'HOOK s':>9} {'HOOK tok/s':>11} {'HOOK PROD':>10} {'V3 s':>9} {'V3 tok/s':>10} {'V3 PROD':>9}")
    print("-" * 86)
    for result in results:
        s = result["summary"]
        print(f"{result['model']:<18} {str(s['avg_hook_seconds']):>9} {str(s['avg_hook_tokens_per_second']):>11} {str(s['hook_production_success_rate']):>10} {str(s['avg_v3_seconds']):>9} {str(s['avg_v3_tokens_per_second']):>10} {str(s['v3_production_success_rate']):>9}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Production-like A/B benchmark for local Ollama models")
    parser.add_argument("video_name", help="e.g. video1")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument("--profile", default="gaming")
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
    print("Qwen A/B benchmark V2 - production config remains unchanged")
    print(f"Video: {args.video_name}")
    print(f"Clips: {len(clips)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Profile: {args.profile}")
    print("Hook: real production Qwen prompt + sanitizer")
    print("V3: real production prompt + system message + normalizer")

    results = [benchmark_model(model=model, clips=clips, transcript=transcript, profile=args.profile) for model in args.models]
    _print_summary(results)

    output_dir = BASE_DIR / "output" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{args.video_name}_qwen_ab_v2_{stamp}.json"
    payload = {"benchmark_version": "qwen-ab-v2-production-like", "video": args.video_name, "profile": args.profile, "tested_clips": len(clips), "models": args.models, "production_config_changed": False, "results": results}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRaport complet: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
