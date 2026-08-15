from pathlib import Path
import hashlib
import json
import shutil
import sys

from faster_whisper import WhisperModel

from src.config import (
    BASE_DIR,
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
)
from src.logger import info, success, warning


model = None
_TRANSCRIPT_CACHE_DIR = BASE_DIR / "cache" / "transcripts"
_TRANSCRIPT_CACHE_VERSION = "whisper-transcript-cache-v1"


def load_model():
    global model
    if model is None:
        info("Se încarcă Whisper...")
        model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
    return model


def _cache_key(video_path: Path) -> str:
    stat = video_path.stat()
    payload = {
        "path": str(video_path.resolve()).lower(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "whisper_model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute": WHISPER_COMPUTE,
        "word_timestamps": True,
        "vad_filter": True,
        "version": _TRANSCRIPT_CACHE_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_path(video_path: Path) -> Path:
    return _TRANSCRIPT_CACHE_DIR / f"{_cache_key(video_path)}.json"


def _valid_transcript(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(payload, list) and bool(payload)
    except Exception:
        return False


def _restore_cache(video_path: Path, output: Path) -> bool:
    try:
        cached = _cache_path(video_path)
    except OSError:
        return False
    if not _valid_transcript(cached):
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, output)
    info(f"[FAST CACHE] Whisper cache hit: {video_path.name}")
    success(f"Transcript restaurat din cache: {output}")
    return True


def _write_cache(video_path: Path, output: Path) -> None:
    try:
        cached = _cache_path(video_path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, cached)
    except Exception as exc:
        warning(f"[FAST CACHE] Nu pot salva transcript cache: {exc}")


def transcribe(video_path: Path):
    video_path = Path(video_path)
    output = TRANSCRIPT_DIR / f"{video_path.stem}.json"

    if _restore_cache(video_path, output):
        return output

    whisper = load_model()
    info(f"Transcriu {video_path.name}")

    segments, _info_data = whisper.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,
    )

    transcript = []
    for segment in segments:
        words = []
        if segment.words:
            for word in segment.words:
                words.append(
                    {
                        "word": word.word.strip(),
                        "start": float(word.start),
                        "end": float(word.end),
                    }
                )

        transcript.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "words": words,
            }
        )

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(transcript, file, indent=4, ensure_ascii=False)

    _write_cache(video_path, output)
    success(f"Transcript salvat: {output}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Utilizare: python src/transcribe.py "video.mp4"')
        sys.exit(1)

    try:
        video = Path(" ".join(sys.argv[1:]))
        transcribe(video)
    except Exception as exc:
        print(exc)
        sys.exit(1)
