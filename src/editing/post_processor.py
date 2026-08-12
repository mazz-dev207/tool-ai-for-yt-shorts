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
RUNTIME_VISUAL_HOOK_MIN_SCORE = 95.0


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


def _plan_path(video_name: str, clip_index: int) -> Path:
    return HIGHLIGHTS_DIR / "v3" / video_name / f"clip_{clip_index}" / "edit_plan.json"


def _load_plan(video_name: str, clip_index: int) -> dict | None:
    path = _plan_path(video_name, clip_index)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _save_plan(video_name: str, clip_index: int, plan: dict) -> None:
    path = _plan_path(video_name, clip_index)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        warning(f"[VISUAL HOOK] nu am putut persista runtime fallback: {exc}")


def _ensure_runtime_visual_hook(plan: dict, clip_index: int) -> bool:
    """Guarantee a conservative visual-hook path during Gemini fallback.

    Normal V3 plans should already contain a visual event selected by the
    Visual Hook Engine. Older/fallback plans can bypass build_edit_plan entirely
    when Gemini quota is exhausted. For those plans only, a very strong gaming
    Hook (>=95) may receive the safe editorial camera-whip effect. We never
    fabricate source-native object/angle/action evidence here.
    """
    if plan.get("visual_events"):
        return False

    metadata = plan.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
        plan["metadata"] = metadata

    profile = str(metadata.get("content_profile", "") or "").strip().lower()
    if profile != "gaming":
        return False

    # Respect an explicit normal-path decision to use no visual hook. The
    # runtime fallback exists for quota/legacy plans that skipped the planner.
    existing_visual_hook = metadata.get("visual_hook")
    fallback_reason = str(metadata.get("fallback_reason", "") or "")
    if isinstance(existing_visual_hook, dict) and not fallback_reason:
        return False

    hook = plan.get("hook") or {}
    try:
        hook_score = float(hook.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        hook_score = 0.0
    if hook_score < RUNTIME_VISUAL_HOOK_MIN_SCORE:
        return False

    direction = "left_to_right" if clip_index % 2 else "right_to_left"
    visual_hook = {
        "technique": "camera_whip",
        "confidence": min(0.78, 0.64 + (hook_score - RUNTIME_VISUAL_HOOK_MIN_SCORE) * 0.035),
        "source_supported": True,
        "source_start": None,
        "source_end": None,
        "apply_mode": "editorial_effect",
        "direction": direction,
        "reason": (
            "Runtime quota fallback: strong gaming Hook score; apply the safe "
            "editorial camera-whip without inventing source actions."
        ),
        "inferred": True,
        "fallback": True,
    }
    plan["visual_events"] = [
        {
            "time": 0.0,
            "effect": "focus_crop",
            "intensity": 0.82,
            "duration": 0.34,
            "target": direction,
            "metadata": {
                "semantic_event": "visual_hook",
                "visual_hook_technique": "camera_whip",
                "direction": direction,
                "fallback": True,
            },
        }
    ]
    metadata["visual_hook"] = visual_hook
    metadata["visual_hook_runtime_fallback"] = True
    plan["no_transformation_needed"] = False
    info(
        f"[VISUAL HOOK][RUNTIME FALLBACK] clip={clip_index} "
        f"technique=camera_whip hook_score={hook_score:.0f} direction={direction}"
    )
    return True


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


def _append_reset(commands: list[tuple[float, str]], end: float) -> None:
    for name, value in (("w", FRAME_WIDTH), ("h", FRAME_HEIGHT), ("x", 0), ("y", 0)):
        commands.append((end, f"{end:.3f} crop@v3 {name} {value};"))


def _append_camera_whip_commands(
    commands: list[tuple[float, str]],
    *,
    start: float,
    duration: float,
    intensity: float,
    direction: str,
) -> None:
    """Approximate a fast camera whip with a short stepped zoomed crop pan.

    The motion is intentionally brief and bounded. It never becomes a persistent
    left/right virtual-camera oscillation and always resets to the stable 9:16
    render after the hook window.
    """
    duration = max(0.18, min(0.50, duration))
    zoom = 1.08 + 0.08 * max(0.0, min(1.0, intensity))
    crop_w = _even(FRAME_WIDTH / zoom)
    crop_h = _even(FRAME_HEIGHT / zoom)
    max_x = max(0, FRAME_WIDTH - crop_w)
    y = max(0, (FRAME_HEIGHT - crop_h) // 2)
    reverse = str(direction or "left_to_right").lower() in {
        "right_to_left", "rtl", "right-left", "left",
    }

    commands.append((start, f"{start:.3f} crop@v3 w {crop_w};"))
    commands.append((start, f"{start:.3f} crop@v3 h {crop_h};"))
    commands.append((start, f"{start:.3f} crop@v3 y {y};"))

    steps = 6
    for index in range(steps):
        progress = index / (steps - 1)
        if reverse:
            progress = 1.0 - progress
        x = int(round(max_x * progress))
        timestamp = start + duration * (index / (steps - 1))
        commands.append((timestamp, f"{timestamp:.3f} crop@v3 x {x};"))

    _append_reset(commands, start + duration)


def _write_visual_commands(plan: dict, clip_index: int) -> Path | None:
    """Create FFmpeg sendcmd commands for safe semantic crop-based effects.

    Standard semantic effects use centered punch-ins. Camera-whip visual hooks
    use a short stepped horizontal crop travel and then return to the stable
    frame. Unsupported effects remain metadata-only.
    """
    commands: list[tuple[float, str]] = []
    for raw in plan.get("visual_events") or []:
        effect = str(raw.get("effect", "")).lower()
        if effect not in EXECUTABLE_VISUAL_EFFECTS:
            continue
        start = max(0.0, float(raw.get("time", 0.0) or 0.0))
        duration = max(0.10, min(1.50, float(raw.get("duration", 0.6) or 0.6)))
        intensity = max(0.0, min(1.0, float(raw.get("intensity", 0.5) or 0.5)))
        metadata = raw.get("metadata") or {}
        visual_hook_technique = str(
            metadata.get("visual_hook_technique", "") or ""
        ).lower()

        if visual_hook_technique == "camera_whip":
            _append_camera_whip_commands(
                commands,
                start=start,
                duration=duration,
                intensity=intensity,
                direction=str(
                    metadata.get("direction")
                    or raw.get("target")
                    or "left_to_right"
                ),
            )
            continue

        zoom = 1.02 + 0.06 * intensity
        crop_w = _even(FRAME_WIDTH / zoom)
        crop_h = _even(FRAME_HEIGHT / zoom)
        x = max(0, (FRAME_WIDTH - crop_w) // 2)
        y = max(0, (FRAME_HEIGHT - crop_h) // 2)
        end = start + duration

        for name, value in (("w", crop_w), ("h", crop_h), ("x", x), ("y", y)):
            commands.append((start, f"{start:.3f} crop@v3 {name} {value};"))
        _append_reset(commands, end)

    if not commands:
        return None
    commands.sort(key=lambda item: item[0])
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMP_DIR / f"clip_{clip_index}_v3_visual.cmd"
    path.write_text("\n".join(line for _time, line in commands) + "\n", encoding="utf-8")
    return path


def _video_filter(
    plan: dict,
    overlay_ass: Path | None,
    visual_commands: Path | None,
) -> str:
    filters: list[str] = []
    if visual_commands is not None:
        command_path = escape_filter_path(visual_commands)
        filters.extend(
            [
                f"sendcmd=f='{command_path}'",
                f"crop@v3=w={FRAME_WIDTH}:h={FRAME_HEIGHT}:x=0:y=0",
                f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            ]
        )
    if overlay_ass is not None:
        filters.append(f"subtitles='{escape_filter_path(overlay_ass)}'")
    return ",".join(filters)


def _dialogue_duck_expression(windows: list[tuple[float, float]]) -> str:
    """Build a conservative output-time dialogue envelope around real SFX events."""
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
        duration = max(0.05, min(2.0, float(raw.get("duration", 0.35) or 0.35)))
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

    # Duck dialogue only while a semantic SFX is active. The reduction is mild
    # (~1.7 dB), so speech remains dominant. Final limiter prevents clipping.
    duck_expression = _dialogue_duck_expression(duck_windows)
    parts.append(
        f"[0:a]volume='{duck_expression}':eval=frame[dialogue]"
    )
    parts.append(
        f"[dialogue]{''.join(labels)}amix=inputs={1 + len(labels)}:"
        "normalize=0:duration=longest,alimiter=limit=0.95[aout]"
    )
    return inputs, ";".join(parts)


def post_process_v3(*, video_name: str, clip_index: int, rendered_path: Path) -> Path:
    plan = _load_plan(video_name, clip_index)
    if not plan:
        return rendered_path

    if V3_ENABLE_SEMANTIC_EFFECTS and _ensure_runtime_visual_hook(plan, clip_index):
        _save_plan(video_name, clip_index, plan)

    overlay_ass = _write_overlay_ass(plan, clip_index)
    visual_commands = (
        _write_visual_commands(plan, clip_index) if V3_ENABLE_SEMANTIC_EFFECTS else None
    )
    vf = _video_filter(plan, overlay_ass, visual_commands)
    sound_inputs, audio_filter = _sound_inputs(plan)
    if not vf and not audio_filter:
        return rendered_path

    temp = TEMP_DIR / f"{rendered_path.stem}_v3.mp4"
    command = ["ffmpeg", "-y", "-i", str(rendered_path), *sound_inputs]
    if audio_filter:
        if vf:
            command.extend(["-vf", vf])
        command.extend(
            ["-filter_complex", audio_filter, "-map", "0:v:0", "-map", "[aout]"]
        )
    elif vf:
        command.extend(["-vf", vf, "-map", "0:v:0", "-map", "0:a?"])
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
        warning(f"[EDIT] semantic post-process failed: {exc}; păstrez render-ul V2.")
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
    return rendered_path
