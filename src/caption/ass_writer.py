from pathlib import Path

from src.caption.utils import ass_time
from src.caption.karaoke import build_word_events
from src.caption.animations import build_animation
from src.caption.styles import CaptionStyle


class ASSWriter:
    def __init__(self, style: CaptionStyle, animation):
        self.style = style
        self.animation = animation

    def write(self, groups, output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as file:
            self.write_header(file)
            self.write_events(file, groups)

    def write_header(self, file):
        style = self.style
        file.write(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n\n"
        )
        file.write("[V4+ Styles]\n\n")
        file.write(
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
            "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
            "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
            "MarginR,MarginV,Encoding\n"
        )
        file.write(
            "Style: "
            f"{style.name},{style.font},{style.font_size},{style.primary_color},"
            f"{style.secondary_color},{style.outline_color},{style.back_color},"
            f"{style.bold},{style.italic},0,0,{style.scale_x},{style.scale_y},"
            f"{style.spacing},{style.angle},{style.border_style},{style.outline},"
            f"{style.shadow},{style.alignment},{style.margin_l},{style.margin_r},"
            f"{style.margin_v},{style.encoding}\n\n"
        )

    def write_events(self, file, groups):
        file.write("[Events]\n\n")
        file.write(
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        )
        animation = build_animation(self.animation)

        for group in groups:
            if not hasattr(group, "words"):
                raise TypeError(f"ASSWriter a primit tip greșit: {type(group)}")
            if not group.words:
                continue

            for event in build_word_events(group):
                start = event["start"]
                end = event["end"]
                text = event["text"]
                if end <= start:
                    continue

                # Smart wrapping stays inside the style's safe margins.
                # Default ASS alignment 5 centers the whole caption block.
                file.write(
                    "Dialogue: 0,"
                    f"{ass_time(start)},{ass_time(end)},{self.style.name},"
                    ",0,0,0,,"
                    f"{{\\q0}}{animation}{text}\n"
                )
