from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

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

# Context local suficient pentru a repara începutul/finalul fără să
# retrimitem inutil peste un minut suplimentar de transcript la fiecare candidat.
RETENTION_CONTEXT_BEFORE = 25.0
RETENTION_CONTEXT_AFTER = 15.0

# Analizăm în continuare cei mai buni 8 candidați, dar generăm doar două
# variante de editare per candidat; de obicei a treia variantă dubla munca.
RETENTION_MAX_CANDIDATES = 8
RETENTION_MAX_VARIANTS = 2

# Editarea extractivă rămâne metoda preferată.
RETENTION_PREFER_ORIGINAL_HOOK = True
RETENTION_MAX_SEGMENTS = 8

# Durată orientativă; optimizer-ul nu forțează 45s.
RETENTION_MIN_CLIP_DURATION = 12.0
RETENTION_MAX_CLIP_DURATION = 60.0
