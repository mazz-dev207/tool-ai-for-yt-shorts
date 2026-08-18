from __future__ import annotations

from src.v3.models import AudioEvent


DEFAULT_SFX_ASSETS = {
    "impact": "impact.wav",
    "accent": "soft_accent.wav",
    "whoosh": "whoosh.wav",
}


def build_audio_events(
    proposal: dict,
    *,
    enabled: bool,
    max_total_events: int = 3,
    gain_cap_db: float = -9.0,
) -> list[AudioEvent]:
    if not enabled:
        return []
    result = []
    for raw in proposal.get("audio_events", []) or []:
        if len(result) >= max_total_events:
            break
        if not isinstance(raw, dict):
            continue
        effect = str(raw.get("effect", "accent") or "accent").lower()
        asset = str(raw.get("asset", "") or "").strip()
        if not asset:
            asset = DEFAULT_SFX_ASSETS.get(effect, "")
        event = AudioEvent(
            time=float(raw.get("time", 0.0) or 0.0),
            effect=effect,
            intensity=float(raw.get("intensity", 0.35) or 0.35),
            duration=float(raw.get("duration", 0.35) or 0.35),
            gain_db=min(
                float(gain_cap_db),
                float(raw.get("gain_db", -12.0) or -12.0),
            ),
            asset=asset,
        )
        if not event.validate():
            result.append(event)
    return result


def audio_safety_warnings(events: list[AudioEvent]) -> list[str]:
    warnings = []
    for event in events:
        if event.gain_db > -6.0:
            warnings.append("audio_effect_gain_too_high")
        if event.duration > 2.0:
            warnings.append("audio_effect_too_long")
    return sorted(set(warnings))
