from pathlib import Path
import subprocess

from src.config import (
    OUTPUT_DIR,
    SUBTITLES_DIR,
    FINAL_DIR,
    TEMP_DIR,
)
from src.logger import info, success
from src.smart_crop import (
    analyze_smart_crop,
    write_sendcmd,
    initial_crop_xy,
)


def escape_filter_path(path: Path) -> str:
    """
    Escape pentru căi Windows folosite în filtre FFmpeg.
    """
    value = path.resolve().as_posix()

    value = value.replace(
        ":",
        r"\:"
    )

    value = value.replace(
        "'",
        r"\'"
    )

    return value


def render(
    clip_name: str
):
    video = (
        OUTPUT_DIR
        / f"{clip_name}.mp4"
    )

    subtitle = (
        SUBTITLES_DIR
        / f"{clip_name}.ass"
    )

    if not video.exists():
        raise FileNotFoundError(
            f"Video lipsă: {video}"
        )

    if not subtitle.exists():
        raise FileNotFoundError(
            f"Subtitle lipsă: {subtitle}"
        )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        FINAL_DIR
        / f"{clip_name}_final.mp4"
    )

    info(
        f"Analizez smart crop pentru {clip_name}"
    )

    crop_plan = analyze_smart_crop(
        video
    )

    command_file = (
        TEMP_DIR
        / f"{clip_name}_crop.cmd"
    )

    write_sendcmd(
        crop_plan,
        command_file
    )

    initial_x, initial_y = initial_crop_xy(
        crop_plan
    )

    subtitle_path = escape_filter_path(
        subtitle
    )

    command_path = escape_filter_path(
        command_file
    )

    #
    # sendcmd controlează crop@smart.
    #
    # crop-ul urmărește fața/subiectul,
    # apoi este scalat exact la 1080x1920.
    #
    filter_chain = (
        f"sendcmd=f='{command_path}',"
        f"crop@smart="
        f"w={crop_plan.crop_width}:"
        f"h={crop_plan.crop_height}:"
        f"x={initial_x}:"
        f"y={initial_y},"
        "scale=1080:1920,"
        f"subtitles='{subtitle_path}'"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        filter_chain,
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "19",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]

    info(
        f"Randez {clip_name} cu smart crop"
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:

        print(
            result.stderr
        )

        raise RuntimeError(
            "FFmpeg a eșuat la smart crop."
        )

    success(
        f"Clip randat: {output.name}"
    )

    return output
