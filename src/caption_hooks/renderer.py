from __future__ import annotations

from pathlib import Path

from src.caption.utils import ass_time
from src.caption_hooks.models import CaptionHookResult
from src.logger import info, warning


_STYLE_NAME = "CaptionHook"
_BOX_STYLE_NAME = "CaptionHookBox"
_PRIMARY = "&H00FFFFFF&"
_ACCENT = "&H0000FFFF&"
_BOX_COLOR = "&H00222222&"

_CARD_LEFT = 105
_CARD_RIGHT = 975
_CARD_TOP_ONE = 145
_CARD_BOTTOM_ONE = 315
_CARD_TOP_TWO = 125
_CARD_BOTTOM_TWO = 355
_CARD_RADIUS = 38


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


def _card_bounds(result: CaptionHookResult) -> tuple[int, int, int, int]:
    two_lines = len([line for line in result.lines if line]) >= 2
    if two_lines:
        return _CARD_LEFT, _CARD_TOP_TWO, _CARD_RIGHT, _CARD_BOTTOM_TWO
    return _CARD_LEFT, _CARD_TOP_ONE, _CARD_RIGHT, _CARD_BOTTOM_ONE


def _rounded_rect(
    left: int,
    top: int,
    right: int,
    bottom: int,
    radius: int = _CARD_RADIUS,
) -> str:
    r = min(radius, (right - left) // 2, (bottom - top) // 2)
    k = int(round(r * 0.5523))
    return (
        f"m {left + r} {top} "
        f"l {right - r} {top} "
        f"b {right - r + k} {top} {right} {top + r - k} {right} {top + r} "
        f"l {right} {bottom - r} "
        f"b {right} {bottom - r + k} {right - r + k} {bottom} {right - r} {bottom} "
        f"l {left + r} {bottom} "
        f"b {left + r - k} {bottom} {left} {bottom - r + k} {left} {bottom - r} "
        f"l {left} {top + r} "
        f"b {left} {top + r - k} {left + r - k} {top} {left + r} {top}"
    )


def _style_lines() -> list[str]:
    return [
        (
            "Style: CaptionHook,Anton,74,&H00FFFFFF,&H0000FFFF,&H00000000,"
            "&H00000000,-1,0,0,0,100,100,0,0,1,3,1,5,95,95,0,1"
        ),
        (
            "Style: CaptionHookBox,Arial,10,&H00222222,&H00222222,&H00222222,"
            "&H00222222,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
        ),
    ]


def _position(result: CaptionHookResult) -> tuple[int, int]:
    # Kept for backwards-compatible unit/debug fixtures. Real generated hooks
    # now use top_safe; webcam layouts are removed at render time.
    if result.position == "gameplay_top":
        return 540, 1160
    left, top, right, bottom = _card_bounds(result)
    return (left + right) // 2, (top + bottom) // 2


def _times(result: CaptionHookResult) -> tuple[float, float]:
    start = max(0.0, float(result.start or 0.05))
    end = start + max(0.2, float(result.duration or 1.4))
    return start, end


def _background_line(result: CaptionHookResult) -> str:
    start, end = _times(result)
    left, top, right, bottom = _card_bounds(result)
    shape = _rounded_rect(left, top, right, bottom)
    tags = (
        rf"{{\an7\pos(0,0)\p1\1c{_BOX_COLOR}\1a&H28&"
        r"\bord0\shad0\fad(80,180)}}"
    )
    return (
        f"Dialogue: 4,{ass_time(start)},{ass_time(end)},{_BOX_STYLE_NAME},"
        f",0,0,0,,{tags}{shape}{{\\p0}}"
    )


def _text_line(result: CaptionHookResult) -> str:
    start, end = _times(result)
    x, y = _position(result)

    animation = (
        rf"{{\an5\pos({x},{y})\q2\fad(80,180)"
        r"\fscx100\fscy100"
        r"\t(0,130,\fscx105\fscy105)"
        r"\t(130,260,\fscx100\fscy100)}}"
    )
    return (
        f"Dialogue: 5,{ass_time(start)},{ass_time(end)},{_STYLE_NAME},"
        f",0,0,0,,{animation}{_styled_text(result)}"
    )


def _is_caption_hook_line(line: str) -> bool:
    if line.startswith(f"Style: {_STYLE_NAME},"):
        return True
    if line.startswith(f"Style: {_BOX_STYLE_NAME},"):
        return True
    return (
        line.startswith("Dialogue:")
        and (
            f",{_STYLE_NAME}," in line
            or f",{_BOX_STYLE_NAME}," in line
        )
    )


def strip_caption_hook_from_ass(source: Path, destination: Path) -> Path:
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
        filtered = [line for line in lines if not _is_caption_hook_line(line)]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        return destination
    except Exception as exc:
        warning(f"[CAPTION HOOK] could not create webcam-safe ASS: {exc}")
        return source


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
        marker = "[Events]"
        if marker not in content:
            raise ValueError("ASS file has no [Events] section")

        additions = []
        if f"Style: {_STYLE_NAME}," not in content:
            additions.append(_style_lines()[0])
        if f"Style: {_BOX_STYLE_NAME}," not in content:
            additions.append(_style_lines()[1])

        if additions:
            content = content.replace(
                marker,
                "\n".join(additions) + "\n\n" + marker,
                1,
            )

        filtered = [
            line
            for line in content.splitlines()
            if not (
                line.startswith("Dialogue:")
                and (
                    f",{_STYLE_NAME}," in line
                    or f",{_BOX_STYLE_NAME}," in line
                )
            )
        ]

        filtered.append(_background_line(result))
        filtered.append(_text_line(result))

        ass_path.write_text(
            "\n".join(filtered) + "\n",
            encoding="utf-8",
        )
        info(f"[CAPTION HOOK] rendered top background card into {ass_path.name}")
        return True

    except Exception as exc:
        warning(
            f"[CAPTION HOOK] ASS render failed: {exc}; "
            "keeping normal subtitles."
        )
        return False
