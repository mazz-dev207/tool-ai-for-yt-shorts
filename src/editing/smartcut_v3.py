from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.config import HIGHLIGHTS_DIR, INPUT_DIR, OUTPUT_DIR, TRANSCRIPT_DIR
from src.logger import info, success, warning
from src.smart_cut import probe_video_duration, refine_edit_plan


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed")


def _encode_args(output: Path) -> list[str]:
    return [
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ]


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _refinement_bounds(role: str) -> tuple[float, float]:
    """Maximum allowed movement outside the AI-authored source range."""
    role = str(role or "context").lower()
    if role in {"cold_open", "hook", "replay", "callback"}:
        return 0.12, 0.18
    if role in {"reaction", "aftermath"}:
        return 0.18, 0.50
    if role == "payoff":
        return 0.18, 0.35
    return 0.18, 0.18


def _normalize_protected_ranges(raw_ranges: list[dict], video_duration: float) -> list[dict]:
    result: list[dict] = []
    for raw in raw_ranges or []:
        if not isinstance(raw, dict):
            continue
        try:
            start = max(0.0, float(raw.get("start")))
            end = min(float(video_duration), float(raw.get("end")))
        except (TypeError, ValueError):
            continue
        if end <= start + 0.03:
            continue
        result.append(
            {
                "start": start,
                "end": end,
                "reason": str(raw.get("reason", "viewer_satisfaction") or "viewer_satisfaction"),
            }
        )
    return result


def _protect_refined_segment(
    *,
    rough_start: float,
    rough_end: float,
    refined_start: float,
    refined_end: float,
    protected_ranges: list[dict],
    video_duration: float,
) -> tuple[float, float, list[str]]:
    """Keep source-grounded tension/comedy/reaction beats inside a V3 segment.

    Only ranges that overlap the AI-authored rough segment are protected. SmartCut
    can still remove real dead air elsewhere.
    """
    start = float(refined_start)
    end = float(refined_end)
    reasons: list[str] = []
    for item in protected_ranges:
        protected_start = float(item["start"])
        protected_end = float(item["end"])
        overlap = min(rough_end, protected_end) - max(rough_start, protected_start)
        if overlap <= 0.02:
            continue
        if start > protected_start:
            start = max(0.0, min(start, protected_start))
            reasons.append(str(item.get("reason", "protected_start")))
        if end < protected_end:
            end = min(float(video_duration), max(end, protected_end))
            reasons.append(str(item.get("reason", "protected_end")))
    return start, end, sorted(set(reasons))


def _refine_segment(
    *,
    video_name: str,
    clip_index: int,
    segment_index: int,
    segment: dict,
    transcript: list[dict],
    video_duration: float,
    protected_ranges: list[dict] | None = None,
) -> dict:
    rough_start = float(segment["start"])
    rough_end = float(segment["end"])
    rough = {"start": rough_start, "end": rough_end}
    role = str(segment.get("role") or "context")
    before_limit, after_limit = _refinement_bounds(role)
    protected_ranges = list(protected_ranges or [])

    try:
        refined = refine_edit_plan(
            video_name=f"{video_name}_v3",
            clip_index=clip_index * 100 + segment_index,
            original_segments=[rough],
            transcript=transcript,
            video_duration=video_duration,
        )
        if refined:
            candidate_start = float(refined[0]["start"])
            candidate_end = float(refined[0]["end"])
            bounded_start = max(
                0.0,
                rough_start - before_limit,
                min(candidate_start, rough_end - 0.05),
            )
            bounded_end = min(
                video_duration,
                rough_end + after_limit,
                max(candidate_end, bounded_start + 0.05),
            )
            bounded_start, bounded_end, protected_reasons = _protect_refined_segment(
                rough_start=rough_start,
                rough_end=rough_end,
                refined_start=bounded_start,
                refined_end=bounded_end,
                protected_ranges=protected_ranges,
                video_duration=video_duration,
            )
            if bounded_end > bounded_start + 0.049:
                if protected_reasons:
                    info(
                        f"[SMARTCUT3][SATISFACTION] protected {role} beat(s): "
                        f"{', '.join(protected_reasons)}"
                    )
                if (
                    abs(bounded_start - candidate_start) > 0.01
                    or abs(bounded_end - candidate_end) > 0.01
                ):
                    info(
                        f"[SMARTCUT3] bounded {role} refinement | "
                        f"AI={rough_start:.2f}->{rough_end:.2f} | "
                        f"SmartCut={candidate_start:.2f}->{candidate_end:.2f} | "
                        f"final={bounded_start:.2f}->{bounded_end:.2f}"
                    )
                return {
                    **segment,
                    "start": round(bounded_start, 4),
                    "end": round(bounded_end, 4),
                }
    except Exception as exc:
        warning(f"[SMARTCUT3] segment refinement failed: {exc}")

    return {
        **segment,
        "start": round(rough_start, 4),
        "end": round(rough_end, 4),
    }


def _cut_segments(video: Path, segments: list[dict], output: Path) -> None:
    clean = []
    for item in segments:
        start = float(item["start"])
        end = float(item["end"])
        if end - start >= 0.05:
            clean.append((start, end))
    if not clean:
        raise ValueError("SmartCut 3.0 timeline fără segmente valide")

    if len(clean) == 1:
        start, end = clean[0]
        _run(
            [
                "ffmpeg", "-y", "-ss", f"{start:.4f}", "-i", str(video),
                "-t", f"{end - start:.4f}", *_encode_args(output),
            ]
        )
        return

    command = ["ffmpeg", "-y"]
    for start, end in clean:
        command.extend(
            [
                "-ss", f"{start:.4f}",
                "-t", f"{end - start:.4f}",
                "-i", str(video),
            ]
        )

    filters = []
    concat = []
    for index in range(len(clean)):
        filters.append(f"[{index}:v:0]setpts=PTS-STARTPTS[v{index}]")
        filters.append(f"[{index}:a:0]asetpts=PTS-STARTPTS[a{index}]")
        concat.append(f"[v{index}][a{index}]")
    filters.append(
        "".join(concat)
        + f"concat=n={len(clean)}:v=1:a=1[outv][outa]"
    )
    command.extend(
        [
            "-filter_complex", ";".join(filters),
            "-map", "[outv]", "-map", "[outa]",
            *_encode_args(output),
        ]
    )
    _run(command)


def cut_v3(video_name: str) -> Path:
    video = INPUT_DIR / f"{video_name}.mp4"
    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"
    if not video.exists():
        raise FileNotFoundError(video)

    clips = _load(highlights_path)
    transcript = _load(transcript_path)
    duration = probe_video_duration(video)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for clip_index, clip in enumerate(clips, start=1):
        raw_segments = clip.get("segments") or [
            {
                "start": float(clip["start"]),
                "end": float(clip["end"]),
                "role": "context",
            }
        ]
        satisfaction = clip.get("satisfaction") if isinstance(clip.get("satisfaction"), dict) else {}
        protected_ranges = _normalize_protected_ranges(
            satisfaction.get("protected_ranges", []) or [],
            duration,
        )
        refined = []
        for segment_index, segment in enumerate(raw_segments, start=1):
            item = {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "role": str(
                    segment.get("role")
                    or segment.get("purpose")
                    or "context"
                ),
            }
            refined.append(
                _refine_segment(
                    video_name=video_name,
                    clip_index=clip_index,
                    segment_index=segment_index,
                    segment=item,
                    transcript=transcript,
                    video_duration=duration,
                    protected_ranges=protected_ranges,
                )
            )

        # IMPORTANT: preserve editorial order. Never sort by source timestamp.
        clip["segments"] = refined
        clip["start"] = round(float(refined[0]["start"]), 3)
        clip["end"] = round(float(refined[-1]["end"]), 3)
        clip["duration"] = round(
            sum(float(item["end"]) - float(item["start"]) for item in refined),
            3,
        )
        info(
            f"[SMARTCUT3] clip={clip_index} segments={len(refined)} "
            f"editorial_order={[item.get('role', 'context') for item in refined]}"
        )
        _cut_segments(
            video,
            refined,
            OUTPUT_DIR / f"clip_{clip_index}.mp4",
        )

    _save(highlights_path, clips)
    success(f"[SMARTCUT3] Exportate {len(clips)} clipuri.")
    return OUTPUT_DIR
