from pathlib import Path
import argparse
import os
import shutil
import sys
import time

from src.config import (
    INPUT_DIR,
    TRANSCRIPT_DIR,
    HIGHLIGHTS_DIR,
    OUTPUT_DIR,
    SUBTITLES_DIR,
    FINAL_DIR,
    TEMP_DIR,
    RETENTION_ENABLED,
    HOOK_OPTIMIZER_ENABLED,
    HIGHLIGHT_MODE,
    CONTENT_PROFILE,
)
from src.logger import info, success
from src.transcribe import transcribe
from src.chunk_transcript import chunk_transcript
from src.highlights.candidate_generator import generate_candidates
from src.highlights.gemini_pipeline import run_gemini_highlight_stage
from src.retention.optimizer import optimize_retention
from src.hooks.optimizer import optimize_hooks
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


def timed_step(func, *args, **kwargs):
    started = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - started


def parse_args():
    parser = argparse.ArgumentParser(description="AI Shorts Pipeline")
    parser.add_argument("video_name", help="Numele fișierului video fără extensia .mp4")
    parser.add_argument(
        "--highlight-mode",
        choices=["legacy", "gemini", "compare"],
        default=HIGHLIGHT_MODE,
        help="legacy, gemini sau compare; implicit vine din .env/config",
    )
    parser.add_argument(
        "--content-profile",
        choices=["auto", "gaming", "entertainment", "podcast", "reaction", "general"],
        default=CONTENT_PROFILE,
        help="Profil folosit de Candidate Discovery, Gemini Judge, Hook Optimizer și SmartCrop.",
    )
    parser.add_argument(
        "--smartcrop-mode",
        choices=["auto", "gameplay-webcam", "gameplay-only"],
        default="auto",
        help=(
            "Politica SmartCrop. gameplay-webcam înseamnă că sursa este cunoscută "
            "ca gameplay + webcam: detectorul caută unde este webcam-ul și forțează "
            "stack 35-40% webcam sus / 60-65% gameplay jos."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_name = args.video_name
    video_path = INPUT_DIR / f"{video_name}.mp4"

    # Runtime-only policy consumed by SmartCrop profile routing during rendering.
    os.environ["SMARTCROP_MODE_RUNTIME"] = args.smartcrop_mode

    if not video_path.exists():
        print(f"Fișierul nu există:\n{video_path}")
        sys.exit(1)

    pipeline_started = time.time()
    timings = {}

    info("Curăț fișierele vechi...")
    clean_started = time.time()
    for folder in [
        TRANSCRIPT_DIR,
        HIGHLIGHTS_DIR,
        OUTPUT_DIR,
        SUBTITLES_DIR,
        FINAL_DIR,
        TEMP_DIR,
    ]:
        clean_folder(folder)
    timings["cleanup"] = time.time() - clean_started

    info("1/8 Transcriere...")
    _, timings["transcription"] = timed_step(transcribe, video_path)

    info("2/8 Creare ferestre analiză...")
    _, timings["chunking"] = timed_step(chunk_transcript, video_name)

    info("3/8 Candidate discovery...")
    _, timings["candidate_discovery"] = timed_step(
        generate_candidates,
        video_name,
        args.highlight_mode in {"gemini", "compare"},
        args.content_profile,
    )
    _, timings["gemini_judge"] = timed_step(
        run_gemini_highlight_stage,
        video_name,
        video_path,
        args.highlight_mode,
        args.content_profile,
    )

    if RETENTION_ENABLED:
        info("4/8 Optimizare pentru retenție...")
        _, timings["retention"] = timed_step(optimize_retention, video_name)
    else:
        info("4/8 Retention engine dezactivat.")
        timings["retention"] = 0.0

    if HOOK_OPTIMIZER_ENABLED:
        info("5/8 Hook START Optimizer...")
        _, timings["hook_optimizer"] = timed_step(
            optimize_hooks,
            video_name,
            video_path,
            args.content_profile,
        )
    else:
        info("5/8 Hook START Optimizer dezactivat; păstrez START-urile existente.")
        timings["hook_optimizer"] = 0.0

    info("6/8 Tăiere / asamblare clipuri...")
    _, timings["cut"] = timed_step(cut, video_name)

    info("7/8 Generare subtitrări...")
    caption_engine = CaptionEngine()
    _, timings["captions"] = timed_step(caption_engine.generate, video_name)

    clips = sorted(
        OUTPUT_DIR.glob("clip_*.mp4"),
        key=lambda path: int(path.stem.split("_")[1]),
    )

    if not clips:
        raise RuntimeError("Nu au fost generate clipuri.")

    info(
        f"8/8 Randare {len(clips)} clipuri... "
        f"SmartCrop mode={args.smartcrop_mode}"
    )
    render_started = time.time()
    for clip in clips:
        render(clip.stem, args.content_profile)
    timings["render"] = time.time() - render_started

    total_elapsed = time.time() - pipeline_started

    print()
    success("Pipeline terminat cu succes!")
    print()
    print(f"Clipuri generate: {len(clips)}")
    print(f"Rezultate finale: {FINAL_DIR}")
    print()
    print("--- TIMPI PIPELINE ---")
    for key in [
        "cleanup",
        "transcription",
        "chunking",
        "candidate_discovery",
        "gemini_judge",
        "retention",
        "hook_optimizer",
        "cut",
        "captions",
        "render",
    ]:
        print(f"{key:20s}: {timings.get(key, 0.0):8.2f}s")
    print(f"{'TOTAL':20s}: {total_elapsed:8.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(f"❌ Eroare: {exc}")
        sys.exit(1)
