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
FORCEABLE_VISUAL_HOOK_EFFECTS = {
    "camera_whip",
    "punch_in",
    "focus_crop",
    "face_zoom",
}
VISUAL_HOOK_STRENGTHS = {"subtle", "medium", "strong"}
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
        warning(f"[VISUAL HOOK] nu am putut persista runtime fallback/override: {exc}")


def normalize_visual_hook_override(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return "auto"
    if normalized not in FORCEABLE_VISUAL_HOOK_EFFECTS:
        allowed = ", ".join(
            sorted(item.replace("_", "-") for item in FORCEABLE_VISUAL_HOOK_EFFECTS)
        )
        raise ValueError(
            f"Visual hook override invalid: {value}. Allowed: auto, {allowed}"
        )
    return normalized


def normalize_visual_hook_strength(value: str | None) -> str:
    normalized = str(value or "medium").strip().lower()
    if normalized not in VISUAL_HOOK_STRENGTHS:
        allowed = ", ".join(sorted(VISUAL_HOOK_STRENGTHS))
        raise ValueError(
            f"Visual hook strength invalid: {value}. Allowed: {allowed}"
        )
    return normalized


def _opening_visual_event(
    effect: str,
    clip_index: int,
    strength: str = "medium",
) -> dict:
    effect = normalize_visual_hook_override(effect)
    strength = normalize_visual_hook_strength(strength)
    direction = "left_to_right" if clip_index % 2 else "right_to_left"

    profiles = {
        "subtle": {
            "camera_whip": (0.72, 0.28, direction),
            "punch_in": (0.68, 0.42, "center"),
            "focus_crop": (0.64, 0.48, "center"),
            "face_zoom": (0.68, 0.42, "face"),
        },
        "medium": {
            "camera_whip": (0.95, 0.42, direction),
            "punch_in": (0.95, 0.68, "center"),
            "focus_crop": (0.90, 0.72, "center"),
            "face_zoom": (0.95, 0.68, "face"),
        },
        "strong": {
            "camera_whip": (1.00, 0.60, direction),
            "punch_in": (1.00, 0.82, "center"),
            "focus_crop": (1.00, 0.90, "center"),
            "face_zoom": (1.00, 0.82, "face"),
        },
    }
    intensity, duration, target = profiles[strength][effect]

    if effect == "camera_whip":
        return {
            "time": 0.0,
            "effect": "focus_crop",
            "intensity": intensity,
            "duration": duration,
            "target": target,
            "metadata": {
                "semantic_event": "visual_hook",
                "visual_hook_technique": "camera_whip",
                "direction": target,
                "visual_hook_strength": strength,
                "cli_forced": True,
            },
        }

    return {
        "time": 0.0,
        "effect": effect,
        "intensity": intensity,
        "duration": duration,
        "target": target,
        "metadata": {
            "semantic_event": "visual_hook",
            "visual_hook_technique": effect,
            "visual_hook_strength": strength,
            "cli_forced": True,
        },
    }


def apply_cli_visual_hook_override(
    plan: dict,
    clip_index: int,
    override: str | None,
    strength: str = "medium",
) -> bool:
    effect = normalize_visual_hook_override(override)
    if effect == "auto":
        return False
    strength = normalize_visual_hook_strength(strength)

    later_events = []
    for raw in plan.get("visual_events") or []:
        try:
            start = float(raw.get("time", 0.0) or 0.0)
        except (TypeError, ValueError):
            start = 0.0
        if start > 0.95:
            later_events.append(raw)

    event = _opening_visual_event(effect, clip_index, strength)
    plan["visual_events"] = [event, *later_events]

    metadata = plan.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
        plan["metadata"] = metadata
    metadata["visual_hook_cli_override"] = effect
    metadata["visual_hook_cli_strength"] = strength
    metadata["visual_hook_cli_forced_all"] = True
    metadata["visual_hook"] = {
        "technique": effect,
        "confidence": 1.0,
        "source_supported": True,
        "source_start": None,
        "source_end": None,
        "apply_mode": "editorial_effect",
        "direction": event.get("target", "center"),
        "strength": strength,
        "reason": "Forced from CLI for every Short in this run.",
        "inferred": False,
        "cli_forced": True,
    }
    plan["no_transformation_needed"] = False
    info(
        f"[VISUAL HOOK][CLI FORCE] clip={clip_index} "
        f"effect={effect} strength={strength} target={event.get('target', 'center')}"
    )
    return True


def _ensure_runtime_visual_hook(plan: dict, clip_index: int) -> bool:
    if plan.get("visual_events"):
        return False

    metadata = plan.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
        plan["metadata"] = metadata

    profile = str(metadata.get("content_profile", "") or "").strip().lower()
    if profile != "gaming":
        return False

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
        "confidence": min(
            0.78,
            0.64 + (hook_score - RUNTIME_VISUAL_HOOK_MIN_SCORE) * 0.035,
        ),
        "source_supported": True,
        "source_start": None,
        "source_end": None,
        "apply_mode": "editorial_effect",
        "direction": direction,
        "strength": "medium",
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
            "intensity": 0.95,
            "duration": 0.42,
            "target": direction,
            "metadata": {
                "semantic_event": "visual_hook",
                "visual_hook_technique": "camera_whip",
                "visual_hook_strength": "medium",
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


# Legacy writer kept only for backward-compatible unit/debug tooling.
# Production rendering below no longer depends on sendcmd.
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


def _effect_geometry(
    effect: str,
    intensity: float,
    strength: str = "medium",
) -> tuple[int, int, str, str]:
    intensity = max(0.0, min(1.0, intensity))
    strength = normalize_visual_hook_strength(strength)
    strength_bonus = {
        "subtle": -0.07,
        "medium": 0.00,
        "strong": 0.22,
    }[strength]

    if effect == "face_zoom":
        zoom = 1.24 + 0.10 * intensity + strength_bonus
        y_ratio = 0.12
    elif effect == "punch_in":
        zoom = 1.18 + 0.10 * intensity + strength_bonus
        y_ratio = 0.50
    elif effect == "focus_crop":
        zoom = 1.14 + 0.08 * intensity + strength_bonus
        y_ratio = 0.50
    else:
        camera_bonus = 0.03 if strength == "subtle" else (0.25 if strength == "strong" else 0.0)
        zoom = 1.22 + 0.08 * intensity + camera_bonus
        y_ratio = 0.50

    zoom = max(1.05, zoom)
    scaled_w = _even(FRAME_WIDTH * zoom)
    scaled_h = _even(FRAME_HEIGHT * zoom)
    max_x = max(0, scaled_w - FRAME_WIDTH)
    max_y = max(0, scaled_h - FRAME_HEIGHT)
    return scaled_w, scaled_h, str(max_x // 2), str(int(round(max_y * y_ratio)))


def _visual_event_filter_parts(plan: dict) -> tuple[list[str], str | None]:
    """Build actual pixel transforms using split/scale/crop/overlay.

    Every effect creates a zoomed branch and overlays it only inside the
    requested time window. Outside that interval the untouched base image is
    shown, so reset behavior is deterministic and does not depend on sendcmd.
    """
    parts: list[str] = []
    current_label = "0:v"
    event_index = 0

    for raw in plan.get("visual_events") or []:
        effect = str(raw.get("effect", "") or "").lower()
        metadata = raw.get("metadata") or {}
        technique = str(metadata.get("visual_hook_technique", "") or "").lower()
        strength = normalize_visual_hook_strength(
            metadata.get("visual_hook_strength", "medium")
        )
        if effect not in EXECUTABLE_VISUAL_EFFECTS:
            continue
        try:
            start = max(0.0, float(raw.get("time", 0.0) or 0.0))
            duration = max(0.12, min(1.50, float(raw.get("duration", 0.6) or 0.6)))
            intensity = max(0.0, min(1.0, float(raw.get("intensity", 0.5) or 0.5)))
        except (TypeError, ValueError):
            continue
        end = start + duration

        semantic_effect = "camera_whip" if technique == "camera_whip" else effect
        scaled_w, scaled_h, x_expr, y_expr = _effect_geometry(
            semantic_effect,
            intensity,
            strength,
        )

        if semantic_effect == "camera_whip":
            max_x = max(0, scaled_w - FRAME_WIDTH)
            direction = str(
                metadata.get("direction")
                or raw.get("target")
                or "left_to_right"
            ).lower()
            progress = f"max(0,min(1,(t-{start:.3f})/{duration:.3f}))"
            if direction in {"right_to_left", "rtl", "right-left", "left"}:
                x_expr = f"{max_x}*(1-{progress})"
            else:
                x_expr = f"{max_x}*{progress}"

        base = f"vh{event_index}base"
        fx = f"vh{event_index}fx"
        cropped = f"vh{event_index}crop"
        out = f"vh{event_index}out"
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
    parts, current_label = _visual_event_filter_parts(plan)

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

    duck_expression = _dialogue_duck_expression(duck_windows)
    parts.append(f"[0:a]volume='{duck_expression}':eval=frame[dialogue]")
    parts.append(
        f"[dialogue]{''.join(labels)}amix=inputs={1 + len(labels)}:"
        "normalize=0:duration=longest,alimiter=limit=0.95[aout]"
    )
    return inputs, ";".join(parts)


def post_process_v3(
    *,
    video_name: str,
    clip_index: int,
    rendered_path: Path,
    visual_hook_override: str = "auto",
    visual_hook_strength: str = "medium",
) -> Path:
    plan = _load_plan(video_name, clip_index)
    if not plan:
        return rendered_path

    changed = False
    if V3_ENABLE_SEMANTIC_EFFECTS:
        if normalize_visual_hook_override(visual_hook_override) != "auto":
            changed = apply_cli_visual_hook_override(
                plan,
                clip_index,
                visual_hook_override,
                visual_hook_strength,
            )
        else:
            changed = _ensure_runtime_visual_hook(plan, clip_index)
        if changed:
            _save_plan(video_name, clip_index, plan)

    overlay_ass = _write_overlay_ass(plan, clip_index)
    video_graph, video_label = (
        _build_video_filtergraph(plan, overlay_ass)
        if V3_ENABLE_SEMANTIC_EFFECTS or overlay_ass is not None
        else (None, None)
    )
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
        if plan.get("visual_events"):
            info(
                f"[EDIT] pixel visual effects rendered for clip_{clip_index}: "
                f"events={len(plan.get('visual_events') or [])}"
            )
        info(f"[EDIT] semantic post-process applied to {rendered_path.name}")
    except Exception as exc:
        warning(f"[EDIT] semantic post-process failed: {exc}; păstrez render-ul V2.")
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
    return rendered_path
