from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import HIGHLIGHTS_DIR


def _load(path: Path, default):
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except Exception:
        return default


def build_ai_usage(video_name: str) -> dict:
    gemini_meta = _load(HIGHLIGHTS_DIR / f"{video_name}_gemini_metadata.json", {})
    highlights = _load(HIGHLIGHTS_DIR / f"{video_name}.json", [])
    hook_meta = _load(HIGHLIGHTS_DIR / f"{video_name}_hook_metadata.json", {})

    selected_shorts = len(highlights) if isinstance(highlights, list) else 0
    gemini_api = int(gemini_meta.get("gemini_api_attempted_count", 0) or 0)
    gemini_cache_hits = int(gemini_meta.get("cache_hits", 0) or 0)
    gemini_failures = int(gemini_meta.get("failures", 0) or 0)
    quota_exhausted = bool(gemini_meta.get("quota_exhausted", False))
    hook_shorts = len(hook_meta.get("shorts", []) or []) if isinstance(hook_meta, dict) else 0

    v3_local = 0
    satisfaction_local = 0
    caption_hook_local = 0
    if isinstance(highlights, list):
        for clip in highlights:
            if not isinstance(clip, dict):
                continue
            if isinstance(clip.get("v3"), dict):
                v3_local += 1
            if isinstance(clip.get("satisfaction"), dict):
                satisfaction_local += 1
            if isinstance(clip.get("caption_hook"), dict):
                caption_hook_local += 1

    return {
        "video": video_name,
        "selected_shorts": selected_shorts,
        "gemini": {
            "role": "final_multimodal_judge_only",
            "api_requests": gemini_api,
            "cache_hits": gemini_cache_hits,
            "failed_requests": gemini_failures,
            "quota_exhausted": quota_exhausted,
            "calls_per_short": round(gemini_api / selected_shorts, 3) if selected_shorts else None,
        },
        "local": {
            "hook_start_provider": "qwen_ollama",
            "hook_start_shorts": hook_shorts,
            "v3_editorial_provider": "qwen_ollama",
            "v3_editorial_shorts": v3_local,
            "satisfaction_provider": "qwen_same_v3_request",
            "satisfaction_shorts": satisfaction_local,
            "caption_hook_provider": "qwen_ollama",
            "caption_hook_shorts": caption_hook_local,
            "gemini_hook_start_calls": 0,
            "gemini_v3_editorial_calls": 0,
            "gemini_satisfaction_calls": 0,
        },
    }


def print_ai_usage(video_name: str) -> dict:
    data = build_ai_usage(video_name)
    gemini = data["gemini"]
    local = data["local"]
    print()
    print("--- AI USAGE / QUOTA ---")
    print(f"Video:                    {data['video']}")
    print(f"Selected Shorts:          {data['selected_shorts']}")
    print(f"Gemini role:              {gemini['role']}")
    print(f"Gemini API requests:      {gemini['api_requests']}")
    print(f"Gemini cache hits:        {gemini['cache_hits']}")
    print(f"Gemini failed requests:   {gemini['failed_requests']}")
    print(f"Gemini quota exhausted:   {gemini['quota_exhausted']}")
    print(f"Gemini calls / Short:     {gemini['calls_per_short']}")
    print("Gemini Hook START calls:  0")
    print("Gemini V3 calls:          0")
    print("Gemini Satisfaction:      0")
    print(f"Hook START:               Qwen local ({local['hook_start_shorts']} Shorts)")
    print(f"V3 Editorial:             Qwen local ({local['v3_editorial_shorts']} Shorts)")
    print(
        "Viewer Satisfaction:      "
        f"Qwen, same V3 request ({local['satisfaction_shorts']} Shorts)"
    )
    return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Utilizare: python -m src.ai_usage video_name")
        raise SystemExit(2)
    print_ai_usage(sys.argv[1])
