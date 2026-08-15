from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src import highlight_selector as legacy_selector
from src.config import BASE_DIR, HIGHLIGHTS_DIR, OLLAMA_MODEL, TEMP_DIR
from src.logger import info, warning


GEMINI_DISCOVERY_MIN_VIRAL_SCORE = 55.0
ENTERTAINMENT_DISCOVERY_MIN_VIRAL_SCORE = 50.0
_DISCOVERY_CACHE_DIR = BASE_DIR / "cache" / "candidate_discovery"
_DISCOVERY_CACHE_VERSION = "candidate-discovery-cache-v1"


def _target_threshold(high_recall: bool, profile: str) -> float:
    if not high_recall:
        return float(legacy_selector.MIN_VIRAL_SCORE)
    if profile == "entertainment":
        return min(
            float(legacy_selector.MIN_VIRAL_SCORE),
            ENTERTAINMENT_DISCOVERY_MIN_VIRAL_SCORE,
        )
    return min(
        float(legacy_selector.MIN_VIRAL_SCORE),
        GEMINI_DISCOVERY_MIN_VIRAL_SCORE,
    )


def _cache_key(video_name: str, high_recall: bool, profile: str) -> str | None:
    chunk_file = TEMP_DIR / f"{video_name}_chunks.json"
    if not chunk_file.exists():
        return None
    try:
        chunk_sha = hashlib.sha256(chunk_file.read_bytes()).hexdigest()
        prompt = legacy_selector.load_selector_prompt()
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    except Exception:
        return None

    payload = {
        "video_name": video_name,
        "chunk_sha256": chunk_sha,
        "prompt_sha256": prompt_sha,
        "profile": profile,
        "high_recall": bool(high_recall),
        "threshold": _target_threshold(high_recall, profile),
        "recovery_threshold": legacy_selector.RECOVERY_MIN_VIRAL_SCORE,
        "recovery_target": legacy_selector.RECOVERY_TARGET_CANDIDATES,
        "min_duration": legacy_selector.MIN_CLIP_DURATION,
        "max_duration": legacy_selector.MAX_CLIP_DURATION,
        "overlap": legacy_selector.OVERLAP_THRESHOLD,
        "ollama_model": OLLAMA_MODEL,
        "num_ctx": legacy_selector.DISCOVERY_NUM_CTX,
        "num_predict": legacy_selector.DISCOVERY_NUM_PREDICT,
        "version": _DISCOVERY_CACHE_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_folder(key: str) -> Path:
    return _DISCOVERY_CACHE_DIR / key


def _restore_cache(video_name: str, key: str | None) -> Path | None:
    if not key:
        return None
    folder = _cache_folder(key)
    cached_highlights = folder / "highlights.json"
    cached_analysis = folder / "analysis.json"
    if not cached_highlights.exists():
        return None
    try:
        payload = json.loads(cached_highlights.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return None
        HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        output = HIGHLIGHTS_DIR / f"{video_name}.json"
        shutil.copy2(cached_highlights, output)
        if cached_analysis.exists():
            shutil.copy2(
                cached_analysis,
                HIGHLIGHTS_DIR / f"{video_name}_analysis.json",
            )
        info(
            f"[FAST CACHE] Candidate Discovery cache hit: "
            f"{len(payload)} candidate(s), 0 Ollama windows rerun."
        )
        return output
    except Exception as exc:
        warning(f"[FAST CACHE] Candidate cache restore failed: {exc}")
        return None


def _store_cache(video_name: str, key: str | None) -> None:
    if not key:
        return
    source = HIGHLIGHTS_DIR / f"{video_name}.json"
    analysis = HIGHLIGHTS_DIR / f"{video_name}_analysis.json"
    if not source.exists():
        return
    try:
        folder = _cache_folder(key)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, folder / "highlights.json")
        if analysis.exists():
            shutil.copy2(analysis, folder / "analysis.json")
    except Exception as exc:
        warning(f"[FAST CACHE] Candidate cache write failed: {exc}")


def generate_candidates(
    video_name: str,
    high_recall: bool = False,
    content_profile: str = "auto",
):
    """Run local Qwen discovery, with deterministic cross-run caching.

    Cache invalidates when transcript chunks, selector prompt, content profile,
    thresholds, Qwen model or relevant discovery settings change.
    """
    profile = str(content_profile or "auto").strip().lower()
    key = _cache_key(video_name, high_recall, profile)
    cached = _restore_cache(video_name, key)
    if cached is not None:
        return cached

    if not high_recall:
        output = legacy_selector.select_highlights(
            video_name,
            content_profile=profile,
            recovery_enabled=False,
        )
        _store_cache(video_name, key)
        return output

    original_threshold = legacy_selector.MIN_VIRAL_SCORE
    target_threshold = GEMINI_DISCOVERY_MIN_VIRAL_SCORE
    if profile == "entertainment":
        target_threshold = ENTERTAINMENT_DISCOVERY_MIN_VIRAL_SCORE

    try:
        legacy_selector.MIN_VIRAL_SCORE = min(
            float(original_threshold),
            target_threshold,
        )
        info(
            f"[HIGHLIGHTS] High-recall candidate mode: "
            f"viral threshold {original_threshold:.0f} -> {legacy_selector.MIN_VIRAL_SCORE:.0f}."
        )
        info(f"[HIGHLIGHTS] Discovery profile: {profile}")
        if profile == "entertainment":
            info("[HIGHLIGHTS] Entertainment discovery includes creator/vlog story beats.")

        output = legacy_selector.select_highlights(
            video_name,
            content_profile=profile,
            recovery_enabled=True,
        )
        _store_cache(video_name, key)
        return output
    finally:
        legacy_selector.MIN_VIRAL_SCORE = original_threshold
