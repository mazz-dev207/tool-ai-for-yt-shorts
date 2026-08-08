import json
import sys
from pathlib import Path

from src.config import (
    TRANSCRIPT_DIR,
    HIGHLIGHTS_DIR,
    SUBTITLES_DIR,
    CAPTION_ANIMATION,
)
from src.logger import info, success
from src.caption.styles import DEFAULT_STYLE
from src.caption.animations import Animation
from src.caption.grouping import group_words, WordGroup, Word
from src.caption.ass_writer import ASSWriter
from src.retention.timeline import extract_remapped_words


class CaptionEngine:
    def __init__(self):
        self.style = DEFAULT_STYLE
        self.animation = getattr(Animation, CAPTION_ANIMATION.upper())
        self.writer = ASSWriter(style=self.style, animation=self.animation)

    def load_json(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def load_transcript(self, video_name: str):
        return self.load_json(TRANSCRIPT_DIR / f"{video_name}.json")

    def load_highlights(self, video_name: str):
        return self.load_json(HIGHLIGHTS_DIR / f"{video_name}.json")

    def extract_words(self, transcript, clip_start, clip_end):
        words = []
        for segment in transcript:
            if segment["end"] < clip_start or segment["start"] > clip_end:
                continue

            for word in segment.get("words", []):
                if word["end"] < clip_start or word["start"] > clip_end:
                    continue

                words.append(
                    {
                        "word": word["word"],
                        "start": float(word["start"]) - clip_start,
                        "end": float(word["end"]) - clip_start,
                    }
                )
        return words

    def validate_groups(self, groups):
        fixed = []
        for group in groups:
            if isinstance(group, WordGroup):
                fixed.append(group)
                continue

            if isinstance(group, list):
                words = []
                for item in group:
                    if isinstance(item, Word):
                        words.append(item)
                    elif isinstance(item, dict):
                        words.append(
                            Word(
                                word=item["word"],
                                start=float(item["start"]),
                                end=float(item["end"]),
                            )
                        )
                if words:
                    fixed.append(
                        WordGroup(words=words, start=words[0].start, end=words[-1].end)
                    )
        return fixed

    def generate_clip(self, transcript, clip, index):
        info(f"Generez subtitrarea {index}")

        if clip.get("segments"):
            words = extract_remapped_words(transcript, clip["segments"])
            info(
                f"Clip {index}: timestamp-urile au fost remapate pentru "
                f"{len(clip['segments'])} segmente extractive."
            )
        else:
            words = self.extract_words(
                transcript,
                float(clip["start"]),
                float(clip["end"]),
            )

        if not words:
            info(f"Clip {index} nu are cuvinte.")
            return

        groups = self.validate_groups(group_words(words))
        info(f"Au fost create {len(groups)} grupuri.")

        if not groups:
            info(f"Clip {index} fără grupuri.")
            return

        output = SUBTITLES_DIR / f"clip_{index}.ass"
        self.writer.write(groups, output)
        success(f"Creat {output.name}")

    def generate(self, video_name: str):
        transcript = self.load_transcript(video_name)
        highlights = self.load_highlights(video_name)
        SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)

        generated = 0
        for index, clip in enumerate(highlights, start=1):
            self.generate_clip(transcript, clip, index)
            generated += 1

        success(f"Generate {generated} fișiere ASS.")
        return SUBTITLES_DIR


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Utilizare: python src/caption_engine.py "video_name"')
        sys.exit(1)

    try:
        CaptionEngine().generate(" ".join(sys.argv[1:]))
    except Exception as exc:
        print(f"Eroare: {exc}")
        sys.exit(1)
