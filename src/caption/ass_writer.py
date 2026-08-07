from pathlib import Path

from src.caption.utils import ass_time
from src.caption.karaoke import build_word_events
from src.caption.animations import build_animation
from src.caption.styles import CaptionStyle


class ASSWriter:


    def __init__(
        self,
        style: CaptionStyle,
        animation
    ):

        self.style = style
        self.animation = animation



    def write(
        self,
        groups,
        output_file: Path
    ):

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:


            self.write_header(
                file
            )


            self.write_events(
                file,
                groups
            )



    def write_header(
        self,
        file
    ):

        style = self.style


        file.write(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n\n"
        )


        file.write(
            "[V4+ Styles]\n\n"
        )


        file.write(
            "Format: "
            "Name,"
            "Fontname,"
            "Fontsize,"
            "PrimaryColour,"
            "SecondaryColour,"
            "OutlineColour,"
            "BackColour,"
            "Bold,"
            "Italic,"
            "Underline,"
            "StrikeOut,"
            "ScaleX,"
            "ScaleY,"
            "Spacing,"
            "Angle,"
            "BorderStyle,"
            "Outline,"
            "Shadow,"
            "Alignment,"
            "MarginL,"
            "MarginR,"
            "MarginV,"
            "Encoding\n"
        )


        file.write(

            "Style: "

            f"{style.name},"

            f"{style.font},"

            f"{style.font_size},"

            f"{style.primary_color},"

            f"{style.secondary_color},"

            f"{style.outline_color},"

            f"{style.back_color},"

            f"{style.bold},"

            f"{style.italic},"

            "0,"

            "0,"

            f"{style.scale_x},"

            f"{style.scale_y},"

            f"{style.spacing},"

            f"{style.angle},"

            f"{style.border_style},"

            f"{style.outline},"

            f"{style.shadow},"

            f"{style.alignment},"

            f"{style.margin_l},"

            f"{style.margin_r},"

            f"{style.margin_v},"

            f"{style.encoding}\n\n"

        )
    def write_events(
        self,
        file,
        groups
    ):

        file.write(
            "[Events]\n\n"
        )

        file.write(
            "Format: "
            "Layer,"
            "Start,"
            "End,"
            "Style,"
            "Name,"
            "MarginL,"
            "MarginR,"
            "MarginV,"
            "Effect,"
            "Text\n"
        )

        animation = build_animation(
            self.animation
        )

        for group in groups:

            # Verificare de siguranță
            if not hasattr(
                group,
                "words"
            ):

                raise TypeError(
                    f"ASSWriter a primit tip greșit: {type(group)}"
                )

            if not group.words:
                continue

            # Generează un event ASS separat pentru fiecare cuvânt.
            # Textul grupului rămâne vizibil, iar doar cuvântul
            # activ este evidențiat de karaoke.py.
            events = build_word_events(
                group
            )

            for event in events:

                start = event["start"]
                end = event["end"]
                text = event["text"]

                # Protecție pentru timestamp-uri invalide.
                if end <= start:
                    continue

                file.write(
                    "Dialogue: 0,"
                    f"{ass_time(start)},"
                    f"{ass_time(end)},"
                    f"{self.style.name},"
                    ",0,0,0,,"
                    f"{animation}{text}\n"
                )
