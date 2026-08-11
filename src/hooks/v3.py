from __future__ import annotations

import re

from src.v3.models import HookPlan


_HIGH_RISK_FACT_TERMS = {
    "brother", "sister", "wife", "husband", "girlfriend", "boyfriend", "boss",
    "employee", "dollar", "dollars", "million", "thousand", "attempt", "attempts",
    "chance", "chances", "last", "final", "first", "only", "nobody", "everyone",
    "impossible", "record", "won", "win", "lost", "lose", "died", "dead", "killed",
    "banned", "fired", "arrested", "regret", "regretted", "wanted", "decided",
    "planned", "intended", "hated", "loved",
}
_GENERIC_CLICKBAIT = (
    "you won't believe", "wait until the end", "craziest thing", "this is insane",
    "watch till the end", "what happens next is crazy",
)


def _clean(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", text or ""))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z']+", str(text or "").lower()))


def _transcript_text(transcript: list[dict], start: float, end: float) -> str:
    parts = []
    for seg in transcript or []:
        try:
            s = float(seg.get("start", 0))
            e = float(seg.get("end", s))
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
        start = float(candidate.get("source_start"))
        end = float(candidate.get("source_end"))
    except Exception:
        return False
    source = _transcript_text(transcript, start - 0.3, end + 0.3).lower()

    def compact(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value).strip()

    return compact(text) in compact(source)


def editorial_text_truth_safe(text: str, context: str) -> bool:
    """Conservative deterministic guard against invented editorial facts.

    Gemini performs the semantic reasoning; this layer blocks high-risk factual
    claims when their factual anchors are absent from the transcript context.
    """
    text = _clean(text)
    context = _clean(context)
    if not text:
        return False
    lowered = text.lower()
    if any(item in lowered for item in _GENERIC_CLICKBAIT):
        return False
    if not _numbers(text).issubset(_numbers(context)):
        return False

    text_tokens = _tokens(text)
    context_tokens = _tokens(context)
    risky_claims = text_tokens & _HIGH_RISK_FACT_TERMS
    if not risky_claims.issubset(context_tokens):
        return False
    return True


def filter_truthful_overlays(raw_overlays: list[dict], context: str) -> list[dict]:
    result = []
    for raw in raw_overlays or []:
        text = _clean(raw.get("text", ""))
        if editorial_text_truth_safe(text, context):
            result.append(dict(raw))
    return result


def select_hook_v3(
    *,
    clip: dict,
    proposal: dict,
    transcript: list[dict],
    context_start: float,
    context_end: float,
    enable_reconstructed: bool = True,
    enable_editorial: bool = True,
) -> HookPlan:
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
                start = float(raw["source_start"])
                end = float(raw["source_end"])
            except Exception:
                continue
            if not (context_start <= start < end <= context_end):
                continue
            options.append(
                HookPlan(
                    mode=mode,
                    score=score,
                    text=_clean(raw.get("text", "")),
                    source_start=start,
                    source_end=end,
                    duration=max(0.1, end - start),
                    reason=_clean(raw.get("reason", ""))[:240],
                    confidence=confidence,
                )
            )

        elif mode == "EDITORIAL" and enable_editorial:
            text = _clean(raw.get("text", ""))
            if not editorial_text_truth_safe(text, context):
                continue
            options.append(
                HookPlan(
                    mode=mode,
                    score=score,
                    text=text,
                    duration=max(0.6, min(2.0, float(raw.get("duration", 1.2) or 1.2))),
                    reason=_clean(raw.get("reason", ""))[:240],
                    confidence=confidence,
                )
            )

    def effective(item: HookPlan) -> float:
        transformation_bonus = {
            "NATIVE": 0,
            "RECONSTRUCTED": 2,
            "EDITORIAL": 3,
        }.get(item.mode, 0)
        return item.score + transformation_bonus + item.confidence * 2

    best = max(options, key=effective)
    if native.score >= 88 and best.mode != "NATIVE" and best.score < native.score + 3:
        return native
    best.validate()
    return best
