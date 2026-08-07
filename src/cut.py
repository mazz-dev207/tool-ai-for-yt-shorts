from pathlib import Path
import subprocess
import json
import sys

from src.config import (
    INPUT_DIR,
    HIGHLIGHTS_DIR,
    OUTPUT_DIR
)

from src.logger import (
    info,
    success
)


def cut(video_name: str):

    video = (
        INPUT_DIR
        /
        f"{video_name}.mp4"
    )

    highlights = (
        HIGHLIGHTS_DIR
        /
        f"{video_name}.json"
    )


    if not video.exists():

        raise FileNotFoundError(
            video
        )


    if not highlights.exists():

        raise FileNotFoundError(
            highlights
        )


    with open(
        highlights,
        encoding="utf-8"
    ) as f:

        clips = json.load(f)



    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    exported = 0


    for index, clip in enumerate(
        clips,
        start=1
    ):


        start = float(
            clip["start"]
        )

        end = float(
            clip["end"]
        )


        duration = end - start


        if duration <= 0:

            info(
                f"Clip {index} ignorat."
            )

            continue



        output = (
            OUTPUT_DIR
            /
            f"clip_{index}.mp4"
        )



        info(
            f"Export clip {index}/{len(clips)}"
        )



        command = [

            "ffmpeg",

            "-y",

            "-ss",
            str(start),

            "-i",
            str(video),

            "-t",
            str(duration),

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

            str(output)

        ]



        result = subprocess.run(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            encoding="utf-8",

            errors="replace"

        )



        if result.returncode != 0:

            raise RuntimeError(

                result.stderr

            )


        exported += 1



    success(
        f"Exportate {exported} clipuri."
    )


    return OUTPUT_DIR





if __name__ == "__main__":


    if len(sys.argv) < 2:

        print(
            'Utilizare: python src/cut.py "video_name"'
        )

        sys.exit(1)



    try:

        video_name = " ".join(
            sys.argv[1:]
        )


        cut(
            video_name
        )


    except Exception as e:

        print(
            f"Eroare: {e}"
        )

        sys.exit(1)