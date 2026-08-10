from dataclasses import dataclass
from dataclasses import replace

from src.config import (
    CAPTION_STYLE,
    FONT_NAME,
    FONT_SIZE,
    FONT_COLOR,
    HIGHLIGHT_COLOR,
    OUTLINE_COLOR,
    BACKGROUND_COLOR,
    OUTLINE,
    SHADOW,
    ALIGNMENT,
    MARGIN_V,
    CAPTION_MARGIN_L,
    CAPTION_MARGIN_R,
)


@dataclass(frozen=True)
class CaptionStyle:
    name: str
    font: str
    font_size: int
    primary_color: str
    secondary_color: str
    outline_color: str
    back_color: str
    bold: int
    italic: int
    outline: int
    shadow: int
    alignment: int
    margin_v: int
    margin_l: int
    margin_r: int
    scale_x: int = 100
    scale_y: int = 100
    spacing: int = 0
    angle: int = 0
    border_style: int = 1
    encoding: int = 1


TIKTOK = CaptionStyle(
    name="Default", font="Anton", font_size=94,
    primary_color="&H00FFFFFF", secondary_color="&H0000FFFF",
    outline_color="&H00000000", back_color="&H50000000",
    bold=-1, italic=0, outline=5, shadow=2, alignment=2,
    margin_v=145, margin_l=70, margin_r=70,
)

REELS = CaptionStyle(
    name="Default", font="Anton", font_size=90,
    primary_color="&H00FFFFFF", secondary_color="&H0000FFFF",
    outline_color="&H00000000", back_color="&H40000000",
    bold=-1, italic=0, outline=4, shadow=1, alignment=2,
    margin_v=150, margin_l=75, margin_r=75,
)

CAPCUT = CaptionStyle(
    name="Default", font="Anton", font_size=92,
    primary_color="&H00FFFFFF", secondary_color="&H0000FFFF",
    outline_color="&H00000000", back_color="&H50000000",
    bold=-1, italic=0, outline=4, shadow=1, alignment=2,
    margin_v=140, margin_l=80, margin_r=80,
)

SUBMAGIC = CaptionStyle(
    name="Default", font="Anton", font_size=96,
    primary_color="&H00FFFFFF", secondary_color="&H0000FFFF",
    outline_color="&H00000000", back_color="&H50000000",
    bold=-1, italic=0, outline=5, shadow=2, alignment=2,
    margin_v=150, margin_l=70, margin_r=70,
)

MINIMAL = CaptionStyle(
    name="Default", font="Montserrat", font_size=72,
    primary_color="&H00FFFFFF", secondary_color="&H00FFFFFF",
    outline_color="&H00000000", back_color="&H00000000",
    bold=0, italic=0, outline=2, shadow=0, alignment=2,
    margin_v=120, margin_l=60, margin_r=60,
)

STYLES = {
    "modern": TIKTOK,
    "tiktok": TIKTOK,
    "reels": REELS,
    "capcut": CAPCUT,
    "submagic": SUBMAGIC,
    "minimal": MINIMAL,
}


def get_style(name: str) -> CaptionStyle:
    if not name:
        return TIKTOK
    return STYLES.get(name.strip().lower(), TIKTOK)


BASE_STYLE = get_style(CAPTION_STYLE)

DEFAULT_STYLE = replace(
    BASE_STYLE,
    font=FONT_NAME,
    font_size=FONT_SIZE,
    primary_color=FONT_COLOR,
    secondary_color=HIGHLIGHT_COLOR,
    outline_color=OUTLINE_COLOR,
    back_color=BACKGROUND_COLOR,
    outline=OUTLINE,
    shadow=SHADOW,
    alignment=ALIGNMENT,
    margin_v=MARGIN_V,
    margin_l=CAPTION_MARGIN_L,
    margin_r=CAPTION_MARGIN_R,
)
