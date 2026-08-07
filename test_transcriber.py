from pathlib import Path

from src.transcribe import transcribe


video = Path(
    "input/video.mp4"
)

transcribe(video)