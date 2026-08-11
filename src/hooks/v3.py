from __future__ import annotations

import re

from src.v3.models import HookPlan


def _clean(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", text or ""))


def _transcript_text(transcript: list[dict], start: float, end: float) -> str:
    parts = []
    for seg in transcript or []:
        try:
            s = float(seg.get("start", 0)); e = float(seg.get("end", s))
        except Exception:
            continue
        if e < start or s > end:
            continue
        text = _clean(seg.get("text", ""))
        if text:
            parts.append(text)
    return " ".join(parts)


def _quote_supported(candidate: dict, transcript: list[dict]) -> bool:
    text = _clean(candidate.get("text", "")).lower()
    if not text:
        return False
    try:
        start = float(candidate.get("source_start")); end = float(candidate.get("source_end"))
    except Exception:
        return False
    source = _transcript_text(transcript, start - 0.3, end + 0.3).lower()
    compact = lambda value: re.sub(r"[^a-z0-9]+", " ", value).strip()
    return compact(text) in compact(source)


def _editorial_truth_safe(text: str, context: str) -> bool:
    text = _clean(text)
    if not text:
        return False
    if not _numbers(text).issubset(_numbers(context)):
        return False
    banned = ("you won't believe", "wait until the end", "craziest thing", "this is insane")
    if any(item in text.lower() for item in banned):
        return False
    return True


def select_hook_v3(*, clip: dict, proposal: dict, transcript: list[dict], context_start: float, context_end: float, enable_reconstructed: bool = True, enable_editorial: bool = True) -> HookPlan:
    native_score = int(max(0, min(100, clip.get("hook_score", 0) or 0)))
    native = HookPlan(
        mode="NATIVE",
        score=native_score,
        source_start=float(clip.get("start", 0.0)),
        reason="Existing Hook Optimizer start.",
        confidence=float(clip.get("hook_confidence", 0.7) or 0.7),
    )
    options = [native]
    context = _transcript_text(transcript, context_start, context_end)

    for raw in proposal.get("hook_candidates", []) or []:
        mode = str(raw.get("mode", "")).upper()
        score = int(max(0, min(100, raw.get("score", 0) or 0)))
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))
        if confidence < 0.50:
            continue
        if mode == "RECONSTRUCTED" and enable_reconstructed:
            if not _quote_supported(raw, transcript):
                continue
            try:
                start = float(raw["source_start"]); end = float(raw["source_end"])
            except Exception:
                continue
            if not (context_start <= start < end <= context_end):
                continue
            options.append(HookPlan(
                mode=mode, score=score, text=_clean(raw.get("text", "")),
                source_start=start, source_end=end, duration=max(0.1, end - start),
                reason=_clean(raw.get("reason", ""))[:240], confidence=confidence,
            ))
        elif mode == "EDITORIAL" and enable_editorial:
            text = _clean(raw.get("text", ""))
            if not _editorial_truth_safe(text, context):
                continue
            options.append(HookPlan(
                mode=mode, score=score, text=text,
                duration=max(0.6, min(2.0, float(raw.get("duration", 1.2) or 1.2))),
                reason=_clean(raw.get("reason", ""))[:240], confidence=confidence,
            ))

    def effective(item: HookPlan) -> float:
        transformation_bonus = {"NATIVE": 0, "RECONSTRUCTED": 2, "EDITORIAL": 3}.get(item.mode, 0)
        return item.score + transformation_bonus + item.confidence * 2

    best = max(options, key=effective)
    if native.score >= 88 and best.mode != "NATIVE" and best.score < native.score + 3:
        return native
    best.validate()
    return best
