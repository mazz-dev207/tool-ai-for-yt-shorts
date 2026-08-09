from __future__ import annotations

from src import highlight_selector as legacy_selector
from src.logger import info


GEMINI_DISCOVERY_MIN_VIRAL_SCORE = 55.0


def generate_candidates(video_name: str, high_recall: bool = False):
    """
    Refolosește selectorul existent ca Candidate Generator.

    legacy mode păstrează pragul original. În Gemini/compare reducem doar temporar
    pragul de discovery pentru recall mai mare; toate celelalte reguli, promptul,
    timestamp snapping și formatul JSON rămân cele existente.
    """
    if not high_recall:
        return legacy_selector.select_highlights(video_name)

    original_threshold = legacy_selector.MIN_VIRAL_SCORE
    try:
        legacy_selector.MIN_VIRAL_SCORE = min(
            float(original_threshold),
            GEMINI_DISCOVERY_MIN_VIRAL_SCORE,
        )
        info(
            f"[HIGHLIGHTS] High-recall candidate mode: "
            f"viral threshold {original_threshold:.0f} -> {legacy_selector.MIN_VIRAL_SCORE:.0f}."
        )
        return legacy_selector.select_highlights(video_name)
    finally:
        legacy_selector.MIN_VIRAL_SCORE = original_threshold
