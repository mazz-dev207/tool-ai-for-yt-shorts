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
    """Maximum allowed movement outside the AI-authored source range.

    The V2 SmartCut engine remains useful for speech boundaries, but V3 Story/EditPlan
    is authoritative. In particular, cold opens must not grow into the full payoff.
    """
    role = str(role or "context").lower()
    if role in {"cold_open", "hook", "replay", "callback"}:
        return 0.12, 0.18
    if role in {"reaction", "aftermath"}:
        return 0.18, 0.50
    if role == "payoff":
        return 0.18, 0.35
    return 0.18, 0.18


def _refine_segment(
    *,
    video_name: str,
    clip_index: int,
    segment_index: int,
    segment: dict,
    transcript: list[dict],
    video_duration: float,
) -> dict:
    rough_start = float(segment["start"])
    rough_end = float(segment["end"])
    rough = {"start": rough_start, "end": rough_end}
    role = str(segment.get("role") or "context")
    before_limit, after_limit = _refinement_bounds(role)

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
            if bounded_end > bounded_start + 0.049:
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
