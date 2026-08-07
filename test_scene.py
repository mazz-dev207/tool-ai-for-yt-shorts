from pathlib import Path

from src.scene_detector import detect_scenes


video = Path(
    "input/video.mp4"
)

detect_scenes(video)