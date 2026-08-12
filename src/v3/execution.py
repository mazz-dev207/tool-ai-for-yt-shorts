from __future__ import annotations

import json

from src.config import HIGHLIGHTS_DIR


def has_executable_v3_changes(video_name: str) -> bool:
    """Return True only when downstream media must be rebuilt.

    Besides editorial timeline/effects, Viewer Satisfaction can reorder/filter
    Shorts or alter END/protected beats. A ranking change is therefore executable
    even when the individual editorial plan otherwise remains native.
    """
    path = HIGHLIGHTS_DIR / f"{video_name}.json"
    if not path.exists():
        return False
    try:
        clips = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(clips, list):
        return False

    for clip in clips:
        v3 = clip.get("v3") or {}
        if not isinstance(v3, dict) or not v3:
            continue
        if bool(v3.get("satisfaction_ranking_changed", False)):
            return True
        satisfaction = clip.get("satisfaction") or {}
        if isinstance(satisfaction, dict) and bool(
            satisfaction.get("end_adjustment_applied", False)
        ):
            return True
        hook_mode = str(v3.get("hook_mode", "NATIVE") or "NATIVE").upper()
        no_transform = bool(v3.get("no_transformation_needed", False))
        if hook_mode != "NATIVE" or not no_transform:
            return True
    return False
