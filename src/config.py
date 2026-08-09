from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent


def _load_local_env() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


_load_local_env()

# Foldere proiect
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
FINAL_DIR = BASE_DIR / "final"
TRANSCRIPT_DIR = BASE_DIR / "transcript"
SCENES_DIR = BASE_DIR / "scenes"
HIGHLIGHTS_DIR = BASE_DIR / "highlights"
SUBTITLES_DIR = BASE_DIR / "subtitles"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"

# Whisper
WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

# Ollama
OLLAMA_MODEL = "qwen3:8b"

# Caption Engine
CAPTION_STYLE = "modern"
CAPTION_ANIMATION = "capcut"
FONT_NAME = "Anton"
FONT_SIZE = 92
FONT_COLOR = "&H00FFFFFF"
HIGHLIGHT_COLOR = "&H0000FFFF"
OUTLINE_COLOR = "&H00000000"
BACKGROUND_COLOR = "&H64000000"
OUTLINE = 4
SHADOW = 1
ALIGNMENT = 2
MARGIN_V = 140
MAX_WORDS = 3
PAUSE_THRESHOLD = 0.35
MIN_WORDS = 1

# Rendering
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

# --------------------------------------------------
# Retention Engine
# --------------------------------------------------
RETENTION_ENABLED = True
RETENTION_CONTEXT_BEFORE = 25.0
RETENTION_CONTEXT_AFTER = 15.0
RETENTION_MAX_CANDIDATES = 8
RETENTION_MAX_VARIANTS = 1
RETENTION_PREFER_ORIGINAL_HOOK = True
RETENTION_MAX_SEGMENTS = 6
RETENTION_MIN_CLIP_DURATION = 12.0
RETENTION_MAX_CLIP_DURATION = 60.0
RETENTION_MIN_FINAL_SCORE = 55

# --------------------------------------------------
# Gemini Highlight Judge
# --------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_ENABLED = _env_bool("GEMINI_ENABLED", True)
HIGHLIGHT_MODE = os.getenv("HIGHLIGHT_MODE", "legacy").strip().lower()
CONTENT_PROFILE = os.getenv("CONTENT_PROFILE", "auto").strip().lower()
GEMINI_CONTEXT_BEFORE = _env_float("GEMINI_CONTEXT_BEFORE", 8.0)
GEMINI_CONTEXT_AFTER = _env_float("GEMINI_CONTEXT_AFTER", 8.0)
GEMINI_MAX_CANDIDATES = _env_int("GEMINI_MAX_CANDIDATES", 60)
GEMINI_TOP_HIGHLIGHTS = _env_int("GEMINI_TOP_HIGHLIGHTS", 10)
GEMINI_MIN_SCORE = _env_int("GEMINI_MIN_SCORE", 55)
GEMINI_OVERLAP_THRESHOLD = _env_float("GEMINI_OVERLAP_THRESHOLD", 0.60)
GEMINI_MAX_RETRIES = _env_int("GEMINI_MAX_RETRIES", 2)
GEMINI_MIN_REFINED_DURATION = _env_float("GEMINI_MIN_REFINED_DURATION", 12.0)
GEMINI_MAX_REFINED_DURATION = _env_float("GEMINI_MAX_REFINED_DURATION", 60.0)
GEMINI_PROMPT_VERSION = "gemini-highlight-v1"
GEMINI_CACHE_DIR = BASE_DIR / "cache" / "gemini_highlights"

# --------------------------------------------------
# SmartCut 2.0
# --------------------------------------------------
SMARTCUT_ENABLED = _env_bool("SMARTCUT_ENABLED", True)
SMARTCUT_DEBUG = _env_bool("SMARTCUT_DEBUG", True)
SMARTCUT_BOUNDARY_SEARCH_WINDOW = _env_float("SMARTCUT_BOUNDARY_SEARCH_WINDOW", 1.25)
SMARTCUT_PRE_CONTEXT = _env_float("SMARTCUT_PRE_CONTEXT", 2.0)
SMARTCUT_POST_CONTEXT = _env_float("SMARTCUT_POST_CONTEXT", 2.0)
SMARTCUT_SENTENCE_PRE_PADDING = _env_float("SMARTCUT_SENTENCE_PRE_PADDING", 0.08)
SMARTCUT_SENTENCE_POST_PADDING = _env_float("SMARTCUT_SENTENCE_POST_PADDING", 0.12)
SMARTCUT_REACTION_SEARCH_WINDOW = _env_float("SMARTCUT_REACTION_SEARCH_WINDOW", 1.50)
SMARTCUT_REACTION_PADDING = _env_float("SMARTCUT_REACTION_PADDING", 0.25)
SMARTCUT_MINIMUM_CUT_SPACING = _env_float("SMARTCUT_MINIMUM_CUT_SPACING", 0.35)
SMARTCUT_MINIMUM_SEGMENT_DURATION = _env_float("SMARTCUT_MINIMUM_SEGMENT_DURATION", 0.80)
SMARTCUT_DEAD_AIR_THRESHOLD = _env_float("SMARTCUT_DEAD_AIR_THRESHOLD", 0.55)

# --------------------------------------------------
# SmartCrop 2.0
# --------------------------------------------------
SMARTCROP_V2_ENABLED = _env_bool("SMARTCROP_V2_ENABLED", True)
SMARTCROP_DEBUG = _env_bool("SMARTCROP_DEBUG", True)
SMARTCROP_DETECT_GAMEPLAY_WEBCAM = _env_bool("SMARTCROP_DETECT_GAMEPLAY_WEBCAM", True)
SMARTCROP_SAMPLE_INTERVAL = _env_float("SMARTCROP_SAMPLE_INTERVAL", 0.50)
SMARTCROP_WEBCAM_PERSISTENCE = _env_float("SMARTCROP_WEBCAM_PERSISTENCE", 0.55)
SMARTCROP_WEBCAM_MIN_AREA_RATIO = _env_float("SMARTCROP_WEBCAM_MIN_AREA_RATIO", 0.004)
SMARTCROP_WEBCAM_MAX_AREA_RATIO = _env_float("SMARTCROP_WEBCAM_MAX_AREA_RATIO", 0.20)
SMARTCROP_WEBCAM_CORNER_ZONE = _env_float("SMARTCROP_WEBCAM_CORNER_ZONE", 0.42)
SMARTCROP_GAMEPLAY_RATIO = _env_float("SMARTCROP_GAMEPLAY_RATIO", 0.72)
SMARTCROP_WEBCAM_RATIO = _env_float("SMARTCROP_WEBCAM_RATIO", 0.28)
SMARTCROP_MOVEMENT_DEAD_ZONE = _env_float("SMARTCROP_MOVEMENT_DEAD_ZONE", 0.06)
SMARTCROP_MAX_CROP_VELOCITY = _env_float("SMARTCROP_MAX_CROP_VELOCITY", 0.18)
SMARTCROP_REACTION_EMPHASIS_ENABLED = _env_bool("SMARTCROP_REACTION_EMPHASIS_ENABLED", True)
SMARTCROP_REACTION_HOLD_DURATION = _env_float("SMARTCROP_REACTION_HOLD_DURATION", 1.40)
SMARTCROP_FALLBACK_TO_EXISTING = _env_bool("SMARTCROP_FALLBACK_TO_EXISTING", True)
