from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.config import HIGHLIGHTS_DIR, TEMP_DIR
from src.v3_config import V3_ENABLE_SEMANTIC_EFFECTS, V3_ENABLE_SOUND_DESIGN, V3_SFX_DIR
from src.logger import info, warning
from src.renderer import escape_filter_path


FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920
EXECUTABLE_VISUAL_EFFECTS = {"punch_in", "face_zoom", "focus_crop"}
DIALOGUE_DUCK_GAIN = 0.82


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
        raise RuntimeError(result.stderr.strip() or "V3 post-process failed")


def _load_plan(video_name: str, clip_index: int) -> dict | None:
    path = HIGHLIGHTS_DIR / "v3" / video_name / f"clip_{clip_index}" / "edit_plan.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _ass_time(value: float) -> str:
    value = max(0.0, float(value))
    hours = int(value // 3600)
    value -= hours * 3600
    minutes = int(value // 60)
    value -= minutes * 60
    return f"{hours}:{minutes:02d}:{value:05.2f}"


def _escape_ass(text: str) -> str:
    return (
        str(text or "")
        .replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", " ")
    )


def _write_overlay_ass(plan: dict, clip_index: int) -> Path | None:
    overlays = plan.get("context_overlays") or []
    if not overlays:
        return None
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMP_DIR / f"clip_{clip_index}_v3_overlay.ass"
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: V3Overlay,Arial,58,&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,3,2,0,8,80,80,165,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for raw in overlays:
        start = float(raw.get("start", 0.1) or 0.1)
        end = start + float(raw.get("duration", 1.2) or 1.2)
        text = _escape_ass(raw.get("text", ""))
        if text:
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},V3Overlay,,0,0,0,,{text}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _even(value: float) -> int:
    parsed = max(2, int(round(value)))
    return parsed if parsed % 2 == 0 else parsed - 1


def _effect_geometry(effect: str, intensity: float) -> tuple[int, int, str, str]:
    """Geometry for generic semantic V3 effects.

    This renderer is intentionally not an opening-hook system. It only executes
    semantic visual events already present in the validated EditPlan.
    """
    intensity = max(0.0, min(1.0, intensity))
    if effect == "face_zoom":
        zoom = 1.14 + 0.08 * intensity
        y_ratio = 0.18
    elif effect == "punch_in":
        zoom = 1.10 + 0.08 * intensity
        y_ratio = 0.50
    else:  # focus_crop
        zoom = 1.08 + 0.07 * intensity
        y_ratio = 0.50

    scaled_w = _even(FRAME_WIDTH * zoom)
    scaled_h = _even(FRAME_HEIGHT * zoom)
    max_x = max(0, scaled_w - FRAME_WIDTH)
    max_y = max(0, scaled_h - FRAME_HEIGHT)
    return (
        scaled_w,
        scaled_h,
        str(max_x // 2),
        str(int(round(max_y * y_ratio))),
    )


def _visual_event_filter_parts(plan: dict) -> tuple[list[str], str | None]:
    parts: list[str] = []
    current_label = "0:v"
    event_index = 0

    for raw in plan.get("visual_events") or []:
        effect = str(raw.get("effect", "") or "").lower()
        if effect not in EXECUTABLE_VISUAL_EFFECTS:
            continue
        try:
            start = max(0.0, float(raw.get("time", 0.0) or 0.0))
            duration = max(
                0.12,
                min(1.50, float(raw.get("duration", 0.6) or 0.6)),
            )
            intensity = max(
                0.0,
                min(1.0, float(raw.get("intensity", 0.5) or 0.5)),
            )
        except (TypeError, ValueError):
            continue

        end = start + duration
        scaled_w, scaled_h, x_expr, y_expr = _effect_geometry(effect, intensity)

        base = f"v3e{event_index}base"
        fx = f"v3e{event_index}fx"
        cropped = f"v3e{event_index}crop"
        out = f"v3e{event_index}out"
        parts.append(f"[{current_label}]split=2[{base}][{fx}]")
        parts.append(
            f"[{fx}]scale={scaled_w}:{scaled_h},"
            f"crop={FRAME_WIDTH}:{FRAME_HEIGHT}:x='{x_expr}':y='{y_expr}'[{cropped}]"
        )
        parts.append(
            f"[{base}][{cropped}]overlay=0:0:"
            f"enable='between(t,{start:.3f},{end:.3f})'[{out}]"
        )
        current_label = out
        event_index += 1

    return parts, current_label if event_index else None


def _build_video_filtergraph(
    plan: dict,
    overlay_ass: Path | None,
) -> tuple[str | None, str | None]:
    parts, current_label = (
        _visual_event_filter_parts(plan)
        if V3_ENABLE_SEMANTIC_EFFECTS
        else ([], None)
    )

    if overlay_ass is not None:
        source = current_label or "0:v"
        overlay_path = escape_filter_path(overlay_ass)
        parts.append(f"[{source}]subtitles='{overlay_path}'[vout]")
        current_label = "vout"

    if not parts or current_label is None:
        return None, None
    return ";".join(parts), current_label


def _dialogue_duck_expression(windows: list[tuple[float, float]]) -> str:
    expression = "1"
    for start, end in reversed(windows):
        expression = (
            f"if(between(t,{start:.3f},{end:.3f}),"
            f"{DIALOGUE_DUCK_GAIN:.3f},{expression})"
        )
    return expression


def _sound_inputs(plan: dict) -> tuple[list[str], str | None]:
    if not V3_ENABLE_SOUND_DESIGN:
        return [], None

    inputs: list[str] = []
    parts: list[str] = []
    labels: list[str] = []
    duck_windows: list[tuple[float, float]] = []
    input_index = 1

    for raw in plan.get("audio_events") or []:
        asset = str(raw.get("asset", "") or "").strip()
        if not asset:
            continue
        path = Path(asset)
        if not path.is_absolute():
            path = V3_SFX_DIR / asset
        if not path.exists():
            warning(f"[SOUND] asset lipsă, omit: {path.name}")
            continue

        start = max(0.0, float(raw.get("time", 0.0) or 0.0))
        duration = max(
            0.05,
            min(2.0, float(raw.get("duration", 0.35) or 0.35)),
        )
        gain = min(-6.0, float(raw.get("gain_db", -12.0) or -12.0))
        delay = int(round(start * 1000))
        inputs.extend(["-i", str(path)])
        label = f"sfx{input_index}"
        parts.append(
            f"[{input_index}:a]volume={gain}dB,adelay={delay}|{delay}[{label}]"
        )
        labels.append(f"[{label}]")
        duck_windows.append((start, start + duration))
        input_index += 1

    if not labels:
        return inputs, None

    duck_expression = _dialogue_duck_expression(duck_windows)
    parts.append(f"[0:a]volume='{duck_expression}':eval=frame[dialogue]")
    parts.append(
        f"[dialogue]{''.join(labels)}amix=inputs={1 + len(labels)}:"
        "normalize=0:duration=longest,alimiter=limit=0.95[aout]"
    )
    return inputs, ";".join(parts)


def post_process_v3(*, video_name: str, clip_index: int, rendered_path: Path) -> Path:
    plan = _load_plan(video_name, clip_index)
    if not plan:
        return rendered_path

    overlay_ass = _write_overlay_ass(plan, clip_index)
    video_graph, video_label = _build_video_filtergraph(plan, overlay_ass)
    sound_inputs, audio_graph = _sound_inputs(plan)

    if not video_graph and not audio_graph:
        return rendered_path

    temp = TEMP_DIR / f"{rendered_path.stem}_v3.mp4"
    command = ["ffmpeg", "-y", "-i", str(rendered_path), *sound_inputs]
    graph_parts = []
    if video_graph:
        graph_parts.append(video_graph)
    if audio_graph:
        graph_parts.append(audio_graph)
    command.extend(["-filter_complex", ";".join(graph_parts)])

    if video_label:
        command.extend(["-map", f"[{video_label}]"])
    else:
        command.extend(["-map", "0:v:0"])
    if audio_graph:
        command.extend(["-map", "[aout]"])
    else:
        command.extend(["-map", "0:a?"])

    command.extend(
        [
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temp),
        ]
    )

    try:
        _run(command)
        temp.replace(rendered_path)
        info(f"[EDIT] semantic post-process applied to {rendered_path.name}")
    except Exception as exc:
        warning(
            f"[EDIT] semantic post-process failed: {exc}; păstrez render-ul V2."
        )
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
    return rendered_path
