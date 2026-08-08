from pathlib import Path
import shutil
import sys

from src.config import (
    INPUT_DIR,
    TRANSCRIPT_DIR,
    HIGHLIGHTS_DIR,
    OUTPUT_DIR,
    SUBTITLES_DIR,
    FINAL_DIR,
    TEMP_DIR,
    RETENTION_ENABLED,
)
from src.logger import info, success
from src.transcribe import transcribe
from src.chunk_transcript import chunk_transcript
from src.highlight_selector import select_highlights
from src.retention.optimizer import optimize_retention
from src.cut import cut
from src.caption_engine import CaptionEngine
from src.renderer import render


def clean_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def main():
    if len(sys.argv) < 2:
        print('Utilizare: python main.py "video_name"')
        sys.exit(1)

    video_name = " ".join(sys.argv[1:])
    video_path = INPUT_DIR / f"{video_name}.mp4"

    if not video_path.exists():
        print(f"Fișierul nu există:\n{video_path}")
        sys.exit(1)

    info("Curăț fișierele vechi...")
    for folder in [
        TRANSCRIPT_DIR,
        HIGHLIGHTS_DIR,
        OUTPUT_DIR,
        SUBTITLES_DIR,
        FINAL_DIR,
        TEMP_DIR,
    ]:
        clean_folder(folder)

    info("1/7 Transcriere...")
    transcribe(video_path)

    info("2/7 Creare ferestre analiză...")
    chunk_transcript(video_name)

    info("3/7 Candidate discovery...")
    select_highlights(video_name)

    if RETENTION_ENABLED:
        info("4/7 Optimizare pentru retenție...")
        optimize_retention(video_name)
    else:
        info("4/7 Retention engine dezactivat.")

    info("5/7 Tăiere / asamblare clipuri...")
    cut(video_name)

    info("6/7 Generare subtitrări...")
    CaptionEngine().generate(video_name)

    clips = sorted(
        OUTPUT_DIR.glob("clip_*.mp4"),
        key=lambda path: int(path.stem.split("_")[1]),
    )

    if not clips:
        raise RuntimeError("Nu au fost generate clipuri.")

    info(f"7/7 Randare {len(clips)} clipuri...")
    for clip in clips:
        render(clip.stem)

    print()
    success("Pipeline terminat cu succes!")
    print()
    print(f"Clipuri generate: {len(clips)}")
    print(f"Rezultate finale: {FINAL_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(f"❌ Eroare: {exc}")
        sys.exit(1)
