from __future__ import annotations

from src.config import (
    SMARTCROP_BLUR_DARKEN,
    SMARTCROP_BLUR_SIGMA,
    VIDEO_WIDTH,
)


def _cover_layer(label: str, output: str, width: int, height: int, blur: bool) -> str:
    filters = (
        f"[{label}]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )
    if blur:
        sigma = max(1.0, float(SMARTCROP_BLUR_SIGMA))
        darken = max(-1.0, min(1.0, float(SMARTCROP_BLUR_DARKEN)))
        filters += f",gblur=sigma={sigma:.2f}:steps=2,eq=brightness={darken:.3f}"
    return f"{filters}[{output}]"


def _foreground_layer(label: str, output: str, width: int, height: int) -> str:
    return (
        f"[{label}]scale={width}:{height}:force_original_aspect_ratio=decrease[{output}]"
    )


def build_gameplay_webcam_stack_filter(
    *,
    plan,
    subtitle_path: str,
    command_path: str,
    initial_x: float,
    initial_y: float,
    blurred_fill: bool = True,
) -> str:
    """Build premium top-webcam / bottom-gameplay composition.

    Both panels always have a full-size content-derived background. The sharp
    foreground is fitted over that background, so no black pad bars are needed.
    """
    webcam = plan.webcam_region
    if webcam is None:
        raise ValueError("GAMEPLAY_WEBCAM_STACK requires webcam_region")

    webcam_height = int(plan.webcam_output_height)
    gameplay_height = int(plan.gameplay_output_height)
    if webcam_height <= 0 or gameplay_height <= 0:
        raise ValueError("Invalid GAMEPLAY_WEBCAM_STACK panel dimensions")

    filters: list[str] = [
        "[0:v]split=2[game_input][cam_input]",
        (
            f"[game_input]sendcmd=f='{command_path}',"
            f"crop@gameplay=w={plan.gameplay_crop_width}:h={plan.gameplay_crop_height}:"
            f"x={initial_x:.3f}:y={initial_y:.3f},split=2[game_bg_src][game_fg_src]"
        ),
        (
            f"[cam_input]crop=w={webcam.w}:h={webcam.h}:x={webcam.x}:y={webcam.y},"
            "split=2[cam_bg_src][cam_fg_src]"
        ),
        _cover_layer("cam_bg_src", "cam_bg", VIDEO_WIDTH, webcam_height, blurred_fill),
        _foreground_layer("cam_fg_src", "cam_fg", VIDEO_WIDTH, webcam_height),
        "[cam_bg][cam_fg]overlay=(W-w)/2:(H-h)/2:shortest=1[cam_panel]",
        _cover_layer("game_bg_src", "game_bg", VIDEO_WIDTH, gameplay_height, blurred_fill),
        _foreground_layer("game_fg_src", "game_fg", VIDEO_WIDTH, gameplay_height),
        "[game_bg][game_fg]overlay=(W-w)/2:(H-h)/2:shortest=1[game_panel]",
        (
            f"[cam_panel][game_panel]vstack=inputs=2,"
            f"subtitles='{subtitle_path}'[vout]"
        ),
    ]
    return ";".join(filters)


def build_safe_cover_stack_filter(
    *,
    plan,
    subtitle_path: str,
    command_path: str,
    initial_x: float,
    initial_y: float,
) -> str:
    """Content-based fallback with no black bars if blurred fill fails."""
    webcam = plan.webcam_region
    if webcam is None:
        raise ValueError("GAMEPLAY_WEBCAM_STACK requires webcam_region")

    webcam_height = int(plan.webcam_output_height)
    gameplay_height = int(plan.gameplay_output_height)
    return ";".join(
        [
            "[0:v]split=2[game_input][cam_input]",
            (
                f"[cam_input]crop=w={webcam.w}:h={webcam.h}:x={webcam.x}:y={webcam.y},"
                f"scale={VIDEO_WIDTH}:{webcam_height}:force_original_aspect_ratio=increase,"
                f"crop={VIDEO_WIDTH}:{webcam_height}[cam_panel]"
            ),
            (
                f"[game_input]sendcmd=f='{command_path}',"
                f"crop@gameplay=w={plan.gameplay_crop_width}:h={plan.gameplay_crop_height}:"
                f"x={initial_x:.3f}:y={initial_y:.3f},"
                f"scale={VIDEO_WIDTH}:{gameplay_height}:force_original_aspect_ratio=increase,"
                f"crop={VIDEO_WIDTH}:{gameplay_height}[game_panel]"
            ),
            (
                f"[cam_panel][game_panel]vstack=inputs=2,"
                f"subtitles='{subtitle_path}'[vout]"
            ),
        ]
    )
