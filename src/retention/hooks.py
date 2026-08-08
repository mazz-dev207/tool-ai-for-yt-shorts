from __future__ import annotations

from typing import Iterable, List, Optional


WEAK_PREFIXES = (
    "știai că",
    "stiai ca",
    "în acest videoclip",
    "in acest videoclip",
    "astăzi vom vorbi",
    "astazi vom vorbi",
    "uite un clip",
    "uită-te până la final",
    "uita-te pana la final",
    "did you know",
    "in this video",
    "watch until the end",
)


def _clamp_score(value) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def normalize_hooks(raw_hooks: Iterable[dict], context_start: float, context_end: float) -> List[dict]:
    result = []

    for hook in raw_hooks or []:
        text = str(hook.get("text", "")).strip()
        if not text:
            continue

        lowered = text.lower()
        generic_penalty = 12 if lowered.startswith(WEAK_PREFIXES) else 0

        generated = bool(hook.get("generated", False))
        normalized = {
            "text": text,
            "type": str(hook.get("type", "unknown")).strip() or "unknown",
            "generated": generated,
            "score": max(0, _clamp_score(hook.get("score")) - generic_penalty),
            "evidence": hook.get("evidence", []),
        }

        if not generated:
            try:
                source_start = float(hook.get("source_start"))
                source_end = float(hook.get("source_end"))
            except (TypeError, ValueError):
                continue

            source_start = max(context_start, min(context_end, source_start))
            source_end = max(context_start, min(context_end, source_end))
            if source_end <= source_start:
                continue

            normalized["source_start"] = source_start
            normalized["source_end"] = source_end
            normalized["application"] = "extractive_audio"
        else:
            # În proiectul actual nu există TTS. Hook-urile generate sunt păstrate
            # pentru ranking/debug și o integrare ulterioară de voice-over.
            normalized["application"] = "metadata_only_no_tts"

        result.append(normalized)

    return result


def select_best_hook(hooks: List[dict], prefer_original: bool = True) -> Optional[dict]:
    if not hooks:
        return None

    def rank(hook: dict):
        generated_penalty = 8 if prefer_original and hook.get("generated") else 0
        return int(hook.get("score", 0)) - generated_penalty

    return max(hooks, key=rank)
