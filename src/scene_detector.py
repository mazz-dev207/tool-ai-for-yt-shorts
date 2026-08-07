from pathlib import Path
import json

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from src.config import SCENES_DIR
from src.logger import info, success


def detect_scenes(video_path: Path):

    info(f"Detect scene pentru: {video_path.name}")

    video = open_video(str(video_path))

    manager = SceneManager()

    manager.add_detector(
        ContentDetector(
            threshold=27.0
        )
    )

    manager.detect_scenes(video)

    scenes = manager.get_scene_list()

    result = []

    for start, end in scenes:
        result.append({
            "start": start.get_seconds(),
            "end": end.get_seconds()
        })


    output = SCENES_DIR / f"{video_path.stem}.json"

    SCENES_DIR.mkdir(exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=4
        )

    success(f"Scene salvate: {output}")

    return output