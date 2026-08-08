from pathlib import Path
import json
import sys

from src.config import (
    TRANSCRIPT_DIR,
    TEMP_DIR,
)
from src.logger import (
    info,
    success,
)


# Fereastra este ZONĂ DE CĂUTARE pentru AI.
# Candidatul final își alege propriile start/end în interior.
WINDOW_SIZE = 60
OVERLAP = 20

STEP = WINDOW_SIZE - OVERLAP


def chunk_transcript(
    video_name: str,
) -> Path:
    transcript_file = (
        TRANSCRIPT_DIR
        / f"{video_name}.json"
    )

    if not transcript_file.exists():
        raise FileNotFoundError(
            transcript_file
        )

    with open(
        transcript_file,
        "r",
        encoding="utf-8",
    ) as file:
        transcript = json.load(
            file
        )

    if not transcript:
        raise ValueError(
            "Transcript gol."
        )

    last_time = float(
        transcript[-1]["end"]
    )

    windows = []

    current_start = 0.0

    while current_start < last_time:
        current_end = min(
            current_start + WINDOW_SIZE,
            last_time,
        )

        segments = []

        for segment in transcript:
            segment_start = float(
                segment["start"]
            )
            segment_end = float(
                segment["end"]
            )

            if (
                segment_end >= current_start
                and segment_start <= current_end
            ):
                segments.append(
                    segment
                )

        if segments:
            windows.append(
                {
                    "start": round(
                        current_start,
                        3,
                    ),
                    "end": round(
                        current_end,
                        3,
                    ),
                    "segments": segments,
                }
            )

        current_start += STEP

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        TEMP_DIR
        / f"{video_name}_chunks.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            windows,
            file,
            indent=4,
            ensure_ascii=False,
        )

    info(
        f"Au fost generate {len(windows)} ferestre "
        f"de până la {WINDOW_SIZE}s "
        f"cu overlap {OVERLAP}s."
    )

    success(
        f"Chunk-uri salvate: {output}"
    )

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Utilizare: python src/chunk_transcript.py "video_name"'
        )
        sys.exit(1)

    try:
        video_name = " ".join(
            sys.argv[1:]
        )

        chunk_transcript(
            video_name
        )

    except Exception as exc:
        print(
            f"Eroare: {exc}"
        )
        sys.exit(1)
