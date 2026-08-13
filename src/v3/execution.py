from __future__ import annotations

import json

from src.config import HIGHLIGHTS_DIR
from src.v3_config import V3_CAPTION_HOOK_ENABLED


def has_executable_v3_changes(video_name: str) -> bool:
    """Return True when downstream media must be rebuilt.

    Besides editorial timeline/effects, Viewer Satisfaction can reorder/filter
    Shorts or alter END/protected beats. Caption Hook is also executable media:
    when enabled for a V3 run, the caption stage must execute so the hook can be
    generated/cached and burned into the ASS subtitle layer before rendering.
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

    has_v3_clip = False
    for clip in clips:
        v3 = clip.get("v3") or {}
        if not isinstance(v3, dict) or not v3:
            continue
        has_v3_clip = True
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

    if V3_CAPTION_HOOK_ENABLED and has_v3_clip:
        return True
    return False
