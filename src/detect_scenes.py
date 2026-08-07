from pathlib import Path
import json
import sys

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

if len(sys.argv) < 2:
    print("Utilizare: python src/detect_scenes.py \"input/video.mp4\"")
    sys.exit(1)

video_path = " ".join(sys.argv[1:])

if not Path(video_path).exists():
    print("Fișierul nu există!")
    sys.exit(1)

print("Analizez scenele...")

video = open_video(video_path)

scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=27.0))

scene_manager.detect_scenes(video)

scene_list = scene_manager.get_scene_list()

results = []

for i, (start, end) in enumerate(scene_list):
    start_sec = start.get_seconds()
    end_sec = end.get_seconds()

    print(f"Scena {i+1}: {start_sec:.2f}s -> {end_sec:.2f}s")

    results.append({
        "start": start_sec,
        "end": end_sec
    })

output = Path("scenes") / (Path(video_path).stem + ".json")
output.parent.mkdir(exist_ok=True)

with open(output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\nScene salvate în:\n{output}")