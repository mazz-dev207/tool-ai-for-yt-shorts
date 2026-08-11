from pathlib import Path
import argparse
import os
import shutil
import sys
import time

from src.config import (
    CONTENT_PROFILE,
    FINAL_DIR,
    HIGHLIGHTS_DIR,
    HIGHLIGHT_MODE,
    HOOK_OPTIMIZER_ENABLED,
    INPUT_DIR,
    OUTPUT_DIR,
    RETENTION_ENABLED,
    SUBTITLES_DIR,
    TEMP_DIR,
    TRANSCRIPT_DIR,
)
from src.v3_config import V3_ENABLED
from src.logger import info, success
from src.transcribe import transcribe
from src.chunk_transcript import chunk_transcript
from src.highlights.candidate_generator import generate_candidates
from src.highlights.gemini_pipeline import run_gemini_highlight_stage
from src.retention.optimizer import optimize_retention
from src.hooks.optimizer import optimize_hooks
from src.v3.pipeline import run_v3_editorial_stage
from src.v3.execution import has_executable_v3_changes
from src.cut import cut
from src.editing.smartcut_v3 import cut_v3
from src.caption_engine import CaptionEngine
from src.renderer import render
from src.editing.post_processor import post_process_v3


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
    parser = argparse.ArgumentParser(description="AI Shorts V3 Editorial Pipeline")
    parser.add_argument(
        "video_name",
        help="Numele fișierului video fără extensia .mp4",
    )
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
        help="Profil folosit de discovery, Gemini, Hook, V3 editorial reasoning și SmartCrop.",
    )
    parser.add_argument(
        "--smartcrop-mode",
        choices=["auto", "gameplay-webcam", "gameplay-only"],
        default="auto",
        help="Politica SmartCrop. gameplay-webcam forțează contractul webcam sus 35-40% / gameplay jos 60-65%.",
    )
    parser.add_argument(
        "--v3-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto folosește V3_ENABLED; on forțează V3; off rulează pipeline-ul V2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_name = args.video_name
    video_path = INPUT_DIR / f"{video_name}.mp4"
    v3_requested = V3_ENABLED if args.v3_mode == "auto" else args.v3_mode == "on"
    os.environ["SMARTCROP_MODE_RUNTIME"] = args.smartcrop_mode

    if not video_path.exists():
        print(f"Fișierul nu există:\n{video_path}")
        sys.exit(1)

    pipeline_started = time.time()
    timings = {}

    info("Curăț fișierele vechi...")
    started = time.time()
    for folder in [
        TRANSCRIPT_DIR,
        HIGHLIGHTS_DIR,
        OUTPUT_DIR,
        SUBTITLES_DIR,
        FINAL_DIR,
        TEMP_DIR,
    ]:
        clean_folder(folder)
    timings["cleanup"] = time.time() - started

    info("1/10 Transcriere...")
    _, timings["transcription"] = timed_step(transcribe, video_path)

    info("2/10 Creare ferestre analiză...")
    _, timings["chunking"] = timed_step(chunk_transcript, video_name)

    info("3/10 Candidate discovery + Highlight Judge...")
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
        info("4/10 Optimizare pentru retenție...")
        _, timings["retention"] = timed_step(optimize_retention, video_name)
    else:
        info("4/10 Retention engine dezactivat.")
        timings["retention"] = 0.0

    if HOOK_OPTIMIZER_ENABLED:
        info("5/10 Hook START Optimizer...")
        _, timings["hook_optimizer"] = timed_step(
            optimize_hooks,
            video_name,
            video_path,
            args.content_profile,
        )
    else:
        info("5/10 Hook START Optimizer dezactivat.")
        timings["hook_optimizer"] = 0.0

    if v3_requested:
        info("6/10 V3 Editorial Angle + Story + Originality + EditPlan...")
        _, timings["v3_editorial"] = timed_step(
            run_v3_editorial_stage,
            video_name,
            video_path,
            args.content_profile,
        )
    else:
        info("6/10 V3 dezactivat; folosesc comportamentul V2.")
        timings["v3_editorial"] = 0.0

    v3_execute = v3_requested and has_executable_v3_changes(video_name)
    if v3_requested and not v3_execute:
        info(
            "[V3] Nicio transformare executabilă necesară; "
            "folosesc exact cut/render path-ul V2."
        )

    info("7/10 SmartCut / asamblare timeline...")
    _, timings["cut"] = timed_step(
        cut_v3 if v3_execute else cut,
        video_name,
    )

    info("8/10 Generare subtitrări + karaoke...")
    caption_engine = CaptionEngine()
    _, timings["captions"] = timed_step(
        caption_engine.generate,
        video_name,
    )

    clips = sorted(
        OUTPUT_DIR.glob("clip_*.mp4"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not clips:
        raise RuntimeError("Nu au fost generate clipuri.")

    info(
        f"9/10 SmartCrop + render {len(clips)} clipuri | "
        f"SmartCrop mode={args.smartcrop_mode}"
    )
    render_started = time.time()
    rendered_outputs = [
        render(clip.stem, args.content_profile)
        for clip in clips
    ]
    timings["render"] = time.time() - render_started

    if v3_execute:
        info("10/10 Semantic visual/audio execution...")
        started = time.time()
        for index, output in enumerate(rendered_outputs, start=1):
            post_process_v3(
                video_name=video_name,
                clip_index=index,
                rendered_path=Path(output),
            )
        timings["semantic_post"] = time.time() - started
    else:
        info("10/10 Semantic V3 effects neutilizate.")
        timings["semantic_post"] = 0.0

    total_elapsed = time.time() - pipeline_started

    print()
    success(
        "AI Shorts V3 pipeline terminat cu succes!"
        if v3_execute
        else "Pipeline terminat cu succes (V2-compatible execution path)!"
    )
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
        "v3_editorial",
        "cut",
        "captions",
        "render",
        "semantic_post",
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
