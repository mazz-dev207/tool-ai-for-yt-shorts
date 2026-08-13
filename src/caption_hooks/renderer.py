from __future__ import annotations

from pathlib import Path

from src.caption.utils import ass_time
from src.caption_hooks.models import CaptionHookResult
from src.logger import info, warning


_STYLE_NAME = "CaptionHook"
_PRIMARY = "&H00FFFFFF"
_ACCENT = "&H0000FFFF"


def _escape_ass(text: str) -> str:
    return (
        str(text or "")
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _styled_text(result: CaptionHookResult) -> str:
    text = r"\N".join(_escape_ass(line) for line in result.lines if line)
    if not text:
        text = _escape_ass(result.text)
    for phrase in (result.emphasis or [])[:1]:
        escaped = _escape_ass(phrase)
        if escaped and escaped in text:
            text = text.replace(
                escaped,
                rf"{{\c{_ACCENT}}}{escaped}{{\c{_PRIMARY}}}",
                1,
            )
    return text


def _position(result: CaptionHookResult) -> tuple[int, int]:
    if result.position == "gameplay_top":
        # GAMEPLAY_WEBCAM_STACK keeps webcam in the upper ~38-48%.
        # Put the editorial hook just inside the gameplay panel, away from face ROI.
        return 540, 790
    return 540, 205


def _style_line() -> str:
    return (
        "Style: CaptionHook,Anton,78,&H00FFFFFF,&H0000FFFF,&H00000000,"
        "&H00000000,-1,0,0,0,100,100,0,0,1,4,1,8,90,90,0,1"
    )


def _dialogue_line(result: CaptionHookResult) -> str:
    x, y = _position(result)
    start = max(0.0, float(result.start or 0.05))
    end = start + max(0.2, float(result.duration or 1.4))
    animation = (
        rf"{{\an8\pos({x},{y})\q2\fad(80,180)"
        r"\fscx100\fscy100"
        r"\t(0,130,\fscx106\fscy106)"
        r"\t(130,260,\fscx100\fscy100)}}"
    )
    return (
        f"Dialogue: 5,{ass_time(start)},{ass_time(end)},{_STYLE_NAME},"
        f",0,0,0,,{animation}{_styled_text(result)}"
    )


def append_caption_hook_to_ass(
    ass_path: Path,
    result: CaptionHookResult,
) -> bool:
    if not result.enabled or not result.text:
        return False
    if not ass_path.exists():
        warning(f"[CAPTION HOOK] ASS file missing, skip overlay: {ass_path}")
        return False

    try:
        content = ass_path.read_text(encoding="utf-8")
        if f"Style: {_STYLE_NAME}," not in content:
            marker = "[Events]"
            if marker not in content:
                raise ValueError("ASS file has no [Events] section")
            content = content.replace(
                marker,
                _style_line() + "\n\n" + marker,
                1,
            )

        # Idempotent reruns: keep all normal/karaoke lines unchanged and replace
        # only a previously generated CaptionHook event.
        filtered = [
            line
            for line in content.splitlines()
            if not (line.startswith("Dialogue: 5,") and f",{_STYLE_NAME}," in line)
        ]
        filtered.append(_dialogue_line(result))
        ass_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        info(f"[CAPTION HOOK] rendered into {ass_path.name}")
        return True
    except Exception as exc:
        warning(f"[CAPTION HOOK] ASS render failed: {exc}; keeping normal subtitles.")
        return False
