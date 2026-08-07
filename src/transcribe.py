from pathlib import Path
import json
import sys

from faster_whisper import WhisperModel

from src.config import (
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE
)

from src.logger import (
    info,
    success
)


model = None


def load_model():

    global model

    if model is None:

        info("Se încarcă Whisper...")

        model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE
        )

    return model


def transcribe(video_path: Path):

    model = load_model()

    info(
        f"Transcriu {video_path.name}"
    )

    segments, info_data = model.transcribe(

        str(video_path),

        word_timestamps=True,

        vad_filter=True

    )

    transcript = []

    for segment in segments:

        words = []

        if segment.words:

            for word in segment.words:

                words.append({

                    "word": word.word.strip(),

                    "start": float(word.start),

                    "end": float(word.end)

                })

        transcript.append({

            "start": float(segment.start),

            "end": float(segment.end),

            "text": segment.text.strip(),

            "words": words

        })

    TRANSCRIPT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        TRANSCRIPT_DIR
        /
        f"{video_path.stem}.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            transcript,
            f,
            indent=4,
            ensure_ascii=False
        )

    success(
        f"Transcript salvat: {output}"
    )

    return output


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            'Utilizare: python src/transcribe.py "video.mp4"'
        )

        sys.exit(1)

    try:

        video = Path(
            " ".join(sys.argv[1:])
        )

        transcribe(video)

    except Exception as e:

        print(e)

        sys.exit(1)