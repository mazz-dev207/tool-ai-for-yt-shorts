from __future__ import annotations

from src.config import (
    HOOK_MAX_SHIFT_AFTER,
    HOOK_MAX_SHIFT_BEFORE,
    HOOK_SEARCH_AFTER,
    HOOK_SEARCH_BEFORE,
    HOOK_SEARCH_STEP,
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def generate_hook_start_candidates(
    original_start: float,
    highlight_end: float,
    video_duration: float,
) -> list[float]:
    """Generate bounded START candidates around the selected highlight start.

    The end of the highlight is not changed here. This stage only proposes semantic
    START positions; duration safety is enforced when the winning START is applied.
    """
    original_start = max(0.0, float(original_start))
    video_duration = max(0.0, float(video_duration))
    highlight_end = min(video_duration, max(original_start, float(highlight_end)))

    before = min(max(0.0, HOOK_SEARCH_BEFORE), max(0.0, HOOK_MAX_SHIFT_BEFORE))
    after = min(max(0.0, HOOK_SEARCH_AFTER), max(0.0, HOOK_MAX_SHIFT_AFTER))
    lower = max(0.0, original_start - before)
    upper = min(highlight_end - 0.35, original_start + after, video_duration)
    if upper < lower:
        return [round(original_start, 3)]

    step = max(0.05, float(HOOK_SEARCH_STEP))
    values: list[float] = []
    current = lower
    while current <= upper + 1e-6:
        values.append(round(current, 3))
        current += step

    values.append(round(_clamp(original_start, lower, upper), 3))
    return sorted(set(values))
