from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json
import sys
import time

import ollama

from src.config import HIGHLIGHTS_DIR, TEMP_DIR, OLLAMA_MODEL
from src.logger import info, success, warning


PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "highlight_selector_prompt.txt"

MIN_VIRAL_SCORE = 70.0
RECOVERY_MIN_VIRAL_SCORE = 40.0
RECOVERY_TARGET_CANDIDATES = 8
MIN_CLIP_DURATION = 15.0
MAX_CLIP_DURATION = 60.0
MAX_CLIPS_PER_WINDOW = 1
OVERLAP_THRESHOLD = 0.35
MAX_BOUNDARY_SNAP_SECONDS = 1.0

DISCOVERY_NUM_CTX = 4096
DISCOVERY_NUM_PREDICT = 700
OLLAMA_KEEP_ALIVE = "30m"

SCORE_KEYS = (
    "hook",
    "retention",
    "entertainment",
    "emotional_intensity",
    "standalone_context",
    "pacing",
    "payoff",
    "viral_potential",
)

VALID_CONTENT_TYPES = {
    "gaming",
    "entertainment",
    "podcast",
    "storytelling",
    "educational",
    "other",
}


def load_selector_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt-ul pentru highlight selector lipsește: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def discovery_profile_guidance(content_profile: str, recovery: bool = False) -> str:
    profile = str(content_profile or "auto").strip().lower()
    guidance = {
        "entertainment": """
ENTERTAINMENT / CREATOR / VLOG DISCOVERY MODE
Treat conversational creator footage as potential story material, not automatic filler.
A vlog candidate can be strong without shouting, danger or a huge reveal.
Look for self-contained micro-stories and memorable beats such as:
- setup -> change -> payoff;
- funny or awkward interactions;
- unexpected answers, reveals or discoveries;
- decisions, mistakes, problems and resolutions;
- emotional reactions or genuine human moments;
- challenges, bets, attempts, wins, fails or escalating situations;
- creator commentary that becomes interesting because something changes;
- travel/location/activity moments only when they contain a clear story beat.
Do NOT reject a moment only because the delivery is calm or conversational.
Prefer a coherent 15-60 second story beat over isolated loudness.
Return content_type=entertainment for vlogs and creator content.
""",
        "gaming": """
GAMING DISCOVERY MODE
Look for clutch plays, fails, surprises, discoveries, funny exchanges, escalating danger,
strategy decisions, reactions and clear gameplay payoffs. Normal uneventful traversal is filler.
""",
        "podcast": """
PODCAST / INTERVIEW DISCOVERY MODE
Look for surprising opinions, concise stories, disagreements, confessions, strong explanations,
funny exchanges, emotional statements and question -> answer payoffs. Calm speech can still be strong.
""",
        "reaction": """
REACTION DISCOVERY MODE
Look for the trigger plus the meaningful reaction and enough context to understand why it matters.
Prefer escalation and payoff over isolated laughter or shouting.
""",
        "general": """
GENERAL DISCOVERY MODE
Favor coherent, self-contained moments with curiosity, progression and payoff.
""",
        "auto": """
AUTO DISCOVERY MODE
Infer the content style from the transcript. If it resembles creator/vlog footage, use vlog logic:
look for micro-stories, interactions, changes, reactions and setup -> payoff rather than only intensity.
""",
    }.get(profile, "")

    if recovery:
        guidance += """

ZERO-CANDIDATE RECOVERY MODE
The first discovery pass found no usable candidates. Favor RECALL now.
Choose the single most promising coherent beat in this window if one exists, even if it is only
moderately viral or needs later trimming. Gemini and Retention will reject weak candidates later.
Return clips: [] only for literal filler, fragmented context, or a window with no coherent event,
interaction, idea, reaction, progression or payoff potential.
"""
    return guidance.strip()


def flatten_words(window: dict) -> list[dict]:
    words: list[dict] = []
    window_start = float(window["start"])
    window_end = float(window["end"])
    for segment in window.get("segments", []):
        for raw_word in segment.get("words", []):
            try:
                start = float(raw_word["start"])
                end = float(raw_word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end < window_start or start > window_end:
                continue
            text = str(raw_word.get("word", "")).strip()
            if text:
                words.append({
                    "word": text,
                    "start": max(window_start, start),
                    "end": min(window_end, max(start, end)),
                })
    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def build_window_context(
    window: dict,
    window_index: int,
    content_profile: str = "auto",
    recovery: bool = False,
) -> str:
    window_start = float(window["start"])
    window_end = float(window["end"])
    transcript_lines = []
    for segment in window.get("segments", []):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        if text:
            transcript_lines.append(f"[{start:.2f}s-{end:.2f}s] {text}")

    mode = "RECOVERY" if recovery else "PRIMARY"
    return f"""
WINDOW #{window_index}
AVAILABLE RANGE: {window_start:.2f}s -> {window_end:.2f}s
DISCOVERY PASS: {mode}
REQUESTED CONTENT PROFILE: {content_profile}

{discovery_profile_guidance(content_profile, recovery)}

Find at most ONE candidate inside this range.
The downstream Gemini/Retention stages will perform strict ranking and detailed editing.

TIMESTAMPED TRANSCRIPT:
{chr(10).join(transcript_lines)}
""".strip()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_score(value: Any) -> float:
    parsed = parse_float(value, 0.0)
    return round(clamp(float(parsed or 0.0), 0.0, 100.0), 2)


def normalize_content_type(value: Any) -> str:
    content_type = str(value or "other").strip().lower()
    aliases = {
        "education": "educational",
        "educational content": "educational",
        "streamer": "entertainment",
        "streamers": "entertainment",
        "challenge": "entertainment",
        "vlog": "entertainment",
        "vlogger": "entertainment",
        "creator": "entertainment",
        "creator vlog": "entertainment",
        "interview": "podcast",
    }
    content_type = aliases.get(content_type, content_type)
    return content_type if content_type in VALID_CONTENT_TYPES else "other"


def snap_boundary(target: float, candidates: list[float]) -> float:
    if not candidates:
        return target
    closest = min(candidates, key=lambda value: abs(value - target))
    return closest if abs(closest - target) <= MAX_BOUNDARY_SNAP_SECONDS else target


def normalize_clip(
    raw_clip: dict,
    window: dict,
    window_index: int,
    min_viral_score: float | None = None,
    discovery_mode: str = "primary",
    content_profile: str = "auto",
) -> dict | None:
    if not isinstance(raw_clip, dict):
        return None

    window_start = float(window["start"])
    window_end = float(window["end"])
    start = parse_float(raw_clip.get("start"))
    end = parse_float(raw_clip.get("end"))
    if start is None or end is None:
        return None

    start = clamp(start, window_start, window_end)
    end = clamp(end, window_start, window_end)
    words = flatten_words(window)
    start = snap_boundary(start, [float(word["start"]) for word in words])
    end = snap_boundary(end, [float(word["end"]) for word in words])
    start = clamp(start, window_start, window_end)
    end = clamp(end, window_start, window_end)
    duration = end - start
    if duration < MIN_CLIP_DURATION or duration > MAX_CLIP_DURATION:
        return None

    raw_scores = raw_clip.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    scores = {key: normalize_score(raw_scores.get(key)) for key in SCORE_KEYS}
    threshold = MIN_VIRAL_SCORE if min_viral_score is None else float(min_viral_score)
    if scores["viral_potential"] < threshold:
        return None

    return {
        "rank": 0,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "title": str(raw_clip.get("title", "Untitled")).strip() or "Untitled",
        "hook": str(raw_clip.get("hook", "")).strip(),
        "payoff": str(raw_clip.get("payoff", "")).strip(),
        "reason": str(raw_clip.get("reason", "")).strip(),
        "scores": scores,
        "discovery_mode": discovery_mode,
        "discovery_profile": str(content_profile or "auto").strip().lower(),
        "_window_index": window_index,
    }


def parse_ai_response(content: str) -> dict:
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen nu a returnat JSON valid.") from exc
    if not isinstance(result, dict):
        raise ValueError("Răspunsul AI trebuie să fie un obiect JSON.")
    clips = result.get("clips", [])
    if clips is None:
        clips = []
    if not isinstance(clips, list):
        raise ValueError('Câmpul "clips" trebuie să fie o listă.')
    result["clips"] = clips[:MAX_CLIPS_PER_WINDOW]
    result["content_type"] = normalize_content_type(result.get("content_type", "other"))
    result["video_summary"] = ""
    return result


def analyze_window(
    window: dict,
    window_index: int,
    selector_prompt: str,
    content_profile: str = "auto",
    recovery: bool = False,
) -> dict:
    context = build_window_context(window, window_index, content_profile, recovery)
    profile_prompt = discovery_profile_guidance(content_profile, recovery)
    effective_system_prompt = f"{selector_prompt}\n\n{profile_prompt}".strip()
    start_time = time.time()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        stream=False,
        think=False,
        keep_alive=OLLAMA_KEEP_ALIVE,
        messages=[
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": context},
        ],
        format="json",
        options={
            "temperature": 0.12 if recovery else 0.10,
            "num_ctx": DISCOVERY_NUM_CTX,
            "num_predict": DISCOVERY_NUM_PREDICT,
        },
    )

    elapsed = time.time() - start_time
    load_ms = float(response.get("load_duration", 0) or 0) / 1_000_000
    prompt_tokens = int(response.get("prompt_eval_count", 0) or 0)
    output_tokens = int(response.get("eval_count", 0) or 0)
    info(
        f"Ollama răspuns în {elapsed:.2f}s | load={load_ms:.0f}ms | "
        f"in={prompt_tokens} tok | out={output_tokens} tok"
    )

    result = parse_ai_response(response["message"]["content"].strip())
    result["_elapsed"] = round(elapsed, 3)
    result["_prompt_tokens"] = prompt_tokens
    result["_output_tokens"] = output_tokens
    return result


def clip_quality_key(clip: dict) -> tuple:
    scores = clip["scores"]
    return (
        scores["viral_potential"],
        scores["retention"],
        scores["payoff"],
        scores["hook"],
        scores["pacing"],
        scores["standalone_context"],
    )


def overlap_ratio(first: dict, second: dict) -> float:
    intersection = max(
        0.0,
        min(float(first["end"]), float(second["end"]))
        - max(float(first["start"]), float(second["start"])),
    )
    if intersection <= 0:
        return 0.0
    first_duration = float(first["end"]) - float(first["start"])
    second_duration = float(second["end"]) - float(second["start"])
    denominator = min(first_duration, second_duration)
    return 0.0 if denominator <= 0 else intersection / denominator


def deduplicate_clips(clips: list[dict]) -> list[dict]:
    ordered = sorted(clips, key=clip_quality_key, reverse=True)
    selected: list[dict] = []
    for candidate in ordered:
        if any(overlap_ratio(candidate, existing) > OVERLAP_THRESHOLD for existing in selected):
            continue
        selected.append(dict(candidate))
    selected.sort(key=clip_quality_key, reverse=True)
    for index, clip in enumerate(selected, start=1):
        clip["rank"] = index
        clip.pop("_window_index", None)
    return selected


def choose_content_type(content_types: list[str]) -> str:
    filtered = [item for item in content_types if item in VALID_CONTENT_TYPES and item != "other"]
    return Counter(filtered).most_common(1)[0][0] if filtered else "other"


def _run_pass(
    windows: list[dict],
    selector_prompt: str,
    content_profile: str,
    recovery: bool,
    min_viral_score: float,
    stop_after: int | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    candidates: list[dict] = []
    debug: list[dict] = []
    content_types: list[str] = []
    pass_name = "RECOVERY" if recovery else "PRIMARY"

    for index, window in enumerate(windows, start=1):
        info(
            f"[{pass_name}] Fereastră {index}/{len(windows)} "
            f"({float(window['start']):.1f}s - {float(window['end']):.1f}s)"
        )
        try:
            result = analyze_window(
                window,
                index,
                selector_prompt,
                content_profile=content_profile,
                recovery=recovery,
            )
            content_types.append(result["content_type"])
            normalized_window_clips = []
            for raw_clip in result["clips"]:
                clip = normalize_clip(
                    raw_clip,
                    window,
                    index,
                    min_viral_score=min_viral_score,
                    discovery_mode="recovery" if recovery else "primary",
                    content_profile=content_profile,
                )
                if clip is None:
                    continue
                normalized_window_clips.append(clip)
                candidates.append(clip)
                scores = clip["scores"]
                info(
                    "Candidat: "
                    f"{clip['start']:.2f}s - {clip['end']:.2f}s | "
                    f"viral={scores['viral_potential']:.0f} | "
                    f"retention={scores['retention']:.0f} | {clip['title']}"
                )

            if not normalized_window_clips:
                info("Fereastra nu conține un candidat care trece filtrele de discovery.")

            debug.append({
                "window_index": index,
                "pass": pass_name.lower(),
                "start": float(window["start"]),
                "end": float(window["end"]),
                "elapsed": result.get("_elapsed", 0.0),
                "prompt_tokens": result.get("_prompt_tokens", 0),
                "output_tokens": result.get("_output_tokens", 0),
                "content_type": result["content_type"],
                "clips": normalized_window_clips,
            })
        except Exception as exc:
            warning(f"Eroare la fereastra {index} ({pass_name.lower()}): {exc}")
            debug.append({
                "window_index": index,
                "pass": pass_name.lower(),
                "start": float(window["start"]),
                "end": float(window["end"]),
                "error": str(exc),
                "clips": [],
            })

        if stop_after is not None and len(deduplicate_clips(candidates)) >= stop_after:
            info(f"[HIGHLIGHTS] Recovery target atins: {stop_after} candidați high-recall.")
            break

    return candidates, debug, content_types


def select_highlights(
    video_name: str,
    content_profile: str = "auto",
    recovery_enabled: bool = False,
):
    chunk_file = TEMP_DIR / f"{video_name}_chunks.json"
    if not chunk_file.exists():
        raise FileNotFoundError(chunk_file)
    with open(chunk_file, "r", encoding="utf-8") as file:
        windows = json.load(file)
    if not isinstance(windows, list):
        raise ValueError("Fișierul de chunk-uri nu conține o listă.")

    profile = str(content_profile or "auto").strip().lower()
    selector_prompt = load_selector_prompt()
    info(f"Analizez {len(windows)} ferestre în modul discovery rapid...")
    info(f"[HIGHLIGHTS] Candidate Discovery profile={profile}")

    discovery_started = time.time()
    candidates, window_debug, content_types = _run_pass(
        windows,
        selector_prompt,
        profile,
        recovery=False,
        min_viral_score=MIN_VIRAL_SCORE,
    )
    final = deduplicate_clips(candidates)

    recovery_used = False
    recovery_debug: list[dict] = []
    recovery_candidates: list[dict] = []
    if not final and recovery_enabled:
        recovery_used = True
        recovery_threshold = min(float(MIN_VIRAL_SCORE), RECOVERY_MIN_VIRAL_SCORE)
        warning(
            "[HIGHLIGHTS] Primary discovery a găsit 0 candidați. "
            f"Pornesc zero-candidate recovery la prag viral {recovery_threshold:.0f}."
        )
        if profile == "entertainment":
            info("[HIGHLIGHTS] Recovery mode: creator/vlog micro-story discovery.")

        recovery_candidates, recovery_debug, recovery_types = _run_pass(
            windows,
            selector_prompt,
            profile,
            recovery=True,
            min_viral_score=recovery_threshold,
            stop_after=RECOVERY_TARGET_CANDIDATES,
        )
        content_types.extend(recovery_types)
        candidates.extend(recovery_candidates)
        final = deduplicate_clips(candidates)
        if final:
            success(f"[HIGHLIGHTS] Recovery a salvat {len(final)} candidați pentru Gemini/Retention.")

    total_elapsed = time.time() - discovery_started
    HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    output = HIGHLIGHTS_DIR / f"{video_name}.json"
    with open(output, "w", encoding="utf-8") as file:
        json.dump(final, file, indent=4, ensure_ascii=False)

    analysis_output = HIGHLIGHTS_DIR / f"{video_name}_analysis.json"
    analysis_payload = {
        "content_type": choose_content_type(content_types),
        "requested_content_profile": profile,
        "recovery_enabled": recovery_enabled,
        "recovery_used": recovery_used,
        "recovery_candidates": len(recovery_candidates),
        "clips": final,
        "windows_analyzed": len(windows),
        "candidates_before_deduplication": len(candidates),
        "selected_clips": len(final),
        "total_discovery_seconds": round(total_elapsed, 3),
        "settings": {
            "min_viral_score": MIN_VIRAL_SCORE,
            "recovery_min_viral_score": RECOVERY_MIN_VIRAL_SCORE,
            "recovery_target_candidates": RECOVERY_TARGET_CANDIDATES,
            "min_clip_duration": MIN_CLIP_DURATION,
            "max_clip_duration": MAX_CLIP_DURATION,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "max_clips_per_window": MAX_CLIPS_PER_WINDOW,
            "num_ctx": DISCOVERY_NUM_CTX,
            "num_predict": DISCOVERY_NUM_PREDICT,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "think": False,
        },
        "windows": window_debug,
        "recovery_windows": recovery_debug,
    }
    with open(analysis_output, "w", encoding="utf-8") as file:
        json.dump(analysis_payload, file, indent=4, ensure_ascii=False)

    if not final:
        warning("Niciun moment nu a trecut nici primary discovery, nici recovery.")

    success(f"Au rămas {len(final)} clipuri.")
    success(f"Candidate discovery terminat în {total_elapsed:.1f}s.")
    success(f"Highlights salvate: {output}")
    success(f"Analiză completă: {analysis_output}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Utilizare: python src/highlight_selector.py "video_name"')
        sys.exit(1)
    try:
        select_highlights(" ".join(sys.argv[1:]))
    except Exception as exc:
        print()
        print(f"Eroare: {exc}")
        sys.exit(1)
