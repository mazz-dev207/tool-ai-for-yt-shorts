from pathlib import Path
import json
import sys

from src.logger import (
    info,
    success
)


WINDOW_SIZE = 45      # secunde
OVERLAP = 15          # secunde

STEP = WINDOW_SIZE - OVERLAP


def chunk_transcript(video_name: str) -> Path:

    transcript_file = (
        Path("transcript")
        /
        f"{video_name}.json"
    )

    if not transcript_file.exists():

        raise FileNotFoundError(
            transcript_file
        )

    with open(
        transcript_file,
        encoding="utf-8"
    ) as f:

        transcript = json.load(f)

    if not transcript:

        raise ValueError(
            "Transcript gol."
        )

    last_time = transcript[-1]["end"]

    windows = []

    current_start = 0

    while current_start < last_time:

        current_end = min(
            current_start + WINDOW_SIZE,
            last_time
        )

        segments = []

        for segment in transcript:

            if (
                segment["end"] >= current_start
                and
                segment["start"] <= current_end
            ):

                segments.append(segment)

        if segments:

            windows.append({

                "start": current_start,

                "end": current_end,

                "segments": segments

            })

        current_start += STEP

    temp = Path("temp")

    temp.mkdir(
        exist_ok=True
    )

    output = (
        temp
        /
        f"{video_name}_chunks.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            windows,
            f,
            indent=4,
            ensure_ascii=False
        )

    info(
        f"Au fost generate {len(windows)} ferestre."
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

    except Exception as e:

        print(
            f"Eroare: {e}"
        )

        sys.exit(1)