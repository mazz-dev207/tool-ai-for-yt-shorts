from __future__ import annotations

from src import highlight_selector as legacy_selector
from src.logger import info


GEMINI_DISCOVERY_MIN_VIRAL_SCORE = 55.0
ENTERTAINMENT_DISCOVERY_MIN_VIRAL_SCORE = 50.0


def generate_candidates(
    video_name: str,
    high_recall: bool = False,
    content_profile: str = "auto",
):
    """Run the local Qwen candidate generator before Gemini/retention.

    In Gemini/compare mode discovery should favor recall. The content profile is
    forwarded to Qwen so entertainment can explicitly recognize creator/vlog
    storytelling instead of treating conversational vlog material as filler.
    """
    profile = str(content_profile or "auto").strip().lower()

    if not high_recall:
        return legacy_selector.select_highlights(
            video_name,
            content_profile=profile,
            recovery_enabled=False,
        )

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

        return legacy_selector.select_highlights(
            video_name,
            content_profile=profile,
            recovery_enabled=True,
        )
    finally:
        legacy_selector.MIN_VIRAL_SCORE = original_threshold
