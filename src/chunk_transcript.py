from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

from src.config import BASE_DIR, TRANSCRIPT_DIR, TEMP_DIR
from src.logger import info, success, warning


# Fereastra este ZONĂ DE CĂUTARE pentru AI.
# Candidatul final își alege propriile start/end în interior.
WINDOW_SIZE = 60
OVERLAP = 20
STEP = WINDOW_SIZE - OVERLAP

_CHUNK_CACHE_DIR = BASE_DIR / "cache" / "transcript_chunks"
_CHUNK_CACHE_VERSION = "transcript-window-cache-v1"
_FAST_CACHE_ENABLED = os.getenv("FAST_CACHE_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on"
}


def _cache_key(transcript_file: Path) -> str:
    digest = hashlib.sha256(transcript_file.read_bytes()).hexdigest()
    payload = {
        "transcript_sha256": digest,
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "version": _CHUNK_CACHE_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_path(transcript_file: Path) -> Path:
    return _CHUNK_CACHE_DIR / f"{_cache_key(transcript_file)}.json"


def _restore_cache(transcript_file: Path, output: Path) -> bool:
    if not _FAST_CACHE_ENABLED:
        return False
    try:
        cached = _cache_path(transcript_file)
        if not cached.exists():
            return False
        payload = json.loads(cached.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            return False
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, output)
        info(f"[FAST CACHE] Transcript chunks cache hit: {output.name}")
        return True
    except Exception:
        return False


def _write_cache(transcript_file: Path, output: Path) -> None:
    if not _FAST_CACHE_ENABLED:
        return
    try:
        cached = _cache_path(transcript_file)
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, cached)
    except Exception as exc:
        warning(f"[FAST CACHE] Nu pot salva chunk cache: {exc}")


def chunk_transcript(video_name: str) -> Path:
    transcript_file = TRANSCRIPT_DIR / f"{video_name}.json"
    if not transcript_file.exists():
        raise FileNotFoundError(transcript_file)

    output = TEMP_DIR / f"{video_name}_chunks.json"
    if _restore_cache(transcript_file, output):
        success(f"Chunk-uri restaurate din cache: {output}")
        return output

    with open(transcript_file, "r", encoding="utf-8") as file:
        transcript = json.load(file)

    if not transcript:
        raise ValueError("Transcript gol.")

    last_time = float(transcript[-1]["end"])
    windows = []
    current_start = 0.0

    while current_start < last_time:
        current_end = min(current_start + WINDOW_SIZE, last_time)
        segments = []

        for segment in transcript:
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])
            if segment_end >= current_start and segment_start <= current_end:
                segments.append(segment)

        if segments:
            windows.append(
                {
                    "start": round(current_start, 3),
                    "end": round(current_end, 3),
                    "segments": segments,
                }
            )

        current_start += STEP

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(windows, file, indent=4, ensure_ascii=False)

    _write_cache(transcript_file, output)
    info(
        f"Au fost generate {len(windows)} ferestre de până la {WINDOW_SIZE}s "
        f"cu overlap {OVERLAP}s."
    )
    success(f"Chunk-uri salvate: {output}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Utilizare: python src/chunk_transcript.py "video_name"')
        sys.exit(1)

    try:
        chunk_transcript(" ".join(sys.argv[1:]))
    except Exception as exc:
        print(f"Eroare: {exc}")
        sys.exit(1)
