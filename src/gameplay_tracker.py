"""Stable gameplay target tracker public API.

The implementation lives in gameplay_tracker_core so the public import path stays
stable for SmartCrop, tests and any external callers.
"""

import src.gameplay_tracker_core as _core


def _source_bonus(source: str) -> float:
    """Rank semantic gameplay targets above generic optical-flow motion.

    `optical_flow_action` is intentionally treated as weak evidence even though
    its name contains the word `action`; otherwise noisy motion can bypass
    target hysteresis after the hold timeout.
    """
    source = str(source or "").lower()
    if "crosshair" in source or "aim" in source:
        return 0.18
    if "player" in source or "character" in source or "vehicle" in source:
        return 0.15
    if "optical" in source:
        return 0.03
    if "enemy" in source or "action" in source or "ball" in source or "puck" in source:
        return 0.10
    return 0.0


# The core resolves this helper at runtime; override it here while preserving the
# stable public module path used by the rest of the project.
_core._source_bonus = _source_bonus

GameplayObservation = _core.GameplayObservation
GameplayTrackPoint = _core.GameplayTrackPoint
GameplayTrackingState = _core.GameplayTrackingState
build_gameplay_observations = _core.build_gameplay_observations
build_stable_gameplay_track = _core.build_stable_gameplay_track
observe_gameplay_motion = _core.observe_gameplay_motion
stabilize_gameplay_observations = _core.stabilize_gameplay_observations

__all__ = [
    "GameplayObservation",
    "GameplayTrackPoint",
    "GameplayTrackingState",
    "build_gameplay_observations",
    "build_stable_gameplay_track",
    "observe_gameplay_motion",
    "stabilize_gameplay_observations",
]
