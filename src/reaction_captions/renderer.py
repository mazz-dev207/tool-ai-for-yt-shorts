from __future__ import annotations

from pathlib import Path

from src.caption.utils import ass_time
from src.logger import info, warning
from src.reaction_captions.models import ReactionEvent


_STYLE_NAME = "ReactionCaption"
_STYLE_LINE = (
    "Style: ReactionCaption,Arial,58,&H00F2F2F2,&H00F2F2F2,&H00000000,"
    "&H00000000,-1,-1,0,0,100,100,0,0,1,3,1,5,90,90,0,1"
)


def _escape_ass(text: str) -> str:
    return (
        str(text or "")
        .replace("\\", r"\\")
        .replace("{", "")
        .replace("}", "")
        .replace("\n", " ")
    )


def _event_line(event: ReactionEvent) -> str:
    start = max(0.0, float(event.start))
    end = max(start + 0.35, float(event.end))
    text = _escape_ass(event.text)
    tags = (
        r"{\an5\pos(540,1480)\q2\fad(80,150)"
        r"\fscx96\fscy96\t(0,120,\fscx102\fscy102)"
        r"\t(120,240,\fscx100\fscy100)}"
    )
    return (
        f"Dialogue: 3,{ass_time(start)},{ass_time(end)},{_STYLE_NAME},"
        f",0,0,0,,{tags}{text}"
    )


def _is_reaction_line(line: str) -> bool:
    if line.startswith(f"Style: {_STYLE_NAME},"):
        return True
    return line.startswith("Dialogue:") and f",{_STYLE_NAME}," in line


def append_reaction_captions_to_ass(
    ass_path: Path,
    events: list[ReactionEvent],
) -> bool:
    """Append a separate non-dialogue reaction layer to an ASS file.

    Idempotent by design: old ReactionCaption style/events are removed before the
    current detection result is written.
    """
    if not ass_path.exists():
        warning(f"[REACTION CAPTION] ASS file missing: {ass_path}")
        return False

    try:
        content = ass_path.read_text(encoding="utf-8")
        if "[Events]" not in content:
            raise ValueError("ASS file has no [Events] section")

        filtered = [
            line
            for line in content.splitlines()
            if not _is_reaction_line(line)
        ]
        content = "\n".join(filtered) + "\n"

        if not events:
            ass_path.write_text(content, encoding="utf-8")
            return False

        content = content.replace(
            "[Events]",
            f"{_STYLE_LINE}\n\n[Events]",
            1,
        )
        lines = content.splitlines()
        lines.extend(_event_line(event) for event in events if event.text)
        ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        info(
            f"[REACTION CAPTION] rendered {len(events)} event(s) into {ass_path.name}"
        )
        return True
    except Exception as exc:
        warning(
            f"[REACTION CAPTION] ASS render failed: {exc}; "
            "normal subtitles remain available."
        )
        return False
