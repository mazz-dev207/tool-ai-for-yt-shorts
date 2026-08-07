from pathlib import Path
import sys

from src.caption_engine import CaptionEngine

from src.logger import (
    info,
    success
)


def generate_subtitles(
    video_name: str
):

    info(
        "Pornesc Caption Engine..."
    )

    engine = CaptionEngine()

    output = engine.generate(
        video_name
    )

    success(
        "Subtitrările au fost generate."
    )

    return output


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            'Utilizare: python src/subtitle_generator.py "video_name"'
        )

        sys.exit(1)

    try:

        generate_subtitles(

            " ".join(
                sys.argv[1:]
            )

        )

    except Exception as e:

        print()

        print(
            f"Eroare: {e}"
        )

        sys.exit(1)