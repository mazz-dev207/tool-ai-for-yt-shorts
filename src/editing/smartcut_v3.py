from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.config import HIGHLIGHTS_DIR, INPUT_DIR, OUTPUT_DIR, TRANSCRIPT_DIR
from src.logger import info, success, warning
from src.smart_cut import probe_video_duration, refine_edit_plan


def _run(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed")


def _encode_args(output: Path) -> list[str]:
    return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _refine_segment(*, video_name: str, clip_index: int, segment_index: int, segment: dict, transcript: list[dict], video_duration: float) -> dict:
    rough = {"start": float(segment["start"]), "end": float(segment["end"])}
    try:
        refined = refine_edit_plan(
            video_name=f"{video_name}_v3", clip_index=clip_index * 100 + segment_index,
            original_segments=[rough], transcript=transcript, video_duration=video_duration,
        )
        if refined:
            return {**segment, "start": float(refined[0]["start"]), "end": float(refined[0]["end"])}
    except Exception as exc:
        warning(f"[SMARTCUT3] segment refinement failed: {exc}")
    return segment


def _cut_segments(video: Path, segments: list[dict], output: Path) -> None:
    clean = []
    for item in segments:
        start = float(item["start"]); end = float(item["end"])
        if end - start >= 0.05:
            clean.append((start, end))
    if not clean:
        raise ValueError("SmartCut 3.0 timeline fără segmente valide")
    if len(clean) == 1:
        start, end = clean[0]
        _run(["ffmpeg", "-y", "-ss", f"{start:.4f}", "-i", str(video), "-t", f"{end - start:.4f}", *_encode_args(output)])
        return

    command = ["ffmpeg", "-y"]
    for start, end in clean:
        command.extend(["-ss", f"{start:.4f}", "-t", f"{end - start:.4f}", "-i", str(video)])
    filters = []
    concat = []
    for index in range(len(clean)):
        filters.append(f"[{index}:v:0]setpts=PTS-STARTPTS[v{index}]")
        filters.append(f"[{index}:a:0]asetpts=PTS-STARTPTS[a{index}]")
        concat.append(f"[v{index}][a{index}]")
    filters.append("".join(concat) + f"concat=n={len(clean)}:v=1:a=1[outv][outa]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[outv]", "-map", "[outa]", *_encode_args(output)])
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
        raw_segments = clip.get("segments") or [{"start": float(clip["start"]), "end": float(clip["end"]), "role": "context"}]
        refined = []
        for segment_index, segment in enumerate(raw_segments, start=1):
            item = {"start": float(segment["start"]), "end": float(segment["end"]), "role": str(segment.get("role") or segment.get("purpose") or "context")}
            refined.append(_refine_segment(
                video_name=video_name, clip_index=clip_index, segment_index=segment_index,
                segment=item, transcript=transcript, video_duration=duration,
            ))

        # IMPORTANT: keep editorial order. Do not sort by source time.
        clip["segments"] = refined
        clip["start"] = round(float(refined[0]["start"]), 3)
        clip["end"] = round(float(refined[-1]["end"]), 3)
        clip["duration"] = round(sum(float(x["end"]) - float(x["start"]) for x in refined), 3)
        info(f"[SMARTCUT3] clip={clip_index} segments={len(refined)} editorial_order={[x.get('role', 'context') for x in refined]}")
        _cut_segments(video, refined, OUTPUT_DIR / f"clip_{clip_index}.mp4")

    _save(highlights_path, clips)
    success(f"[SMARTCUT3] Exportate {len(clips)} clipuri.")
    return OUTPUT_DIR
