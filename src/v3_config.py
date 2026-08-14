from pathlib import Path
import json
import os

from src.config import BASE_DIR


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


V3_ENABLED = _env_bool("V3_ENABLED", True)
V3_ENABLE_ORIGINALITY_ENGINE = _env_bool("ENABLE_ORIGINALITY_ENGINE", True)
V3_ENABLE_STORY_RESTRUCTURING = _env_bool("ENABLE_STORY_RESTRUCTURING", True)
V3_ENABLE_EDITORIAL_HOOKS = _env_bool("ENABLE_EDITORIAL_HOOKS", True)
V3_ENABLE_RECONSTRUCTED_HOOKS = _env_bool("ENABLE_RECONSTRUCTED_HOOKS", True)
V3_ENABLE_CONTEXT_OVERLAYS = _env_bool("ENABLE_CONTEXT_OVERLAYS", True)
V3_ENABLE_SEMANTIC_EFFECTS = _env_bool("ENABLE_SEMANTIC_EFFECTS", True)
V3_ENABLE_SOUND_DESIGN = _env_bool("ENABLE_SOUND_DESIGN", True)
V3_ENABLE_ORIGINALITY_QA = _env_bool("ENABLE_ORIGINALITY_QA", True)

V3_ORIGINALITY_MIN_SCORE = _env_int("ORIGINALITY_MIN_SCORE", 60)
V3_MAX_EDIT_PLAN_RETRIES = _env_int("MAX_EDIT_PLAN_RETRIES", 2)
V3_MAX_EFFECTS_PER_EVENT = _env_int("MAX_EFFECTS_PER_EVENT", 2)
V3_CONTEXT_BEFORE = _env_float("V3_CONTEXT_BEFORE", 8.0)
V3_CONTEXT_AFTER = _env_float("V3_CONTEXT_AFTER", 8.0)
V3_EDITORIAL_PROMPT_VERSION = os.getenv(
    "V3_EDITORIAL_PROMPT_VERSION",
    "ai-shorts-v3-editorial-2-satisfaction",
).strip()
V3_CACHE_DIR = BASE_DIR / "cache" / "v3_editorial"
V3_SFX_DIR = BASE_DIR / "assets" / "sfx"
V3_EDITORIAL_PROFILE_NAME = os.getenv("V3_EDITORIAL_PROFILE", "mazclips").strip().lower()
V3_EDITORIAL_PROFILE_PATH = BASE_DIR / "config" / "editorial_profile.json"

# --------------------------------------------------
# Viewer Satisfaction Engine
# --------------------------------------------------
V3_SATISFACTION_ENABLED = _env_bool("SATISFACTION_ENABLED", True)
V3_SATISFACTION_DEBUG = _env_bool("SATISFACTION_DEBUG", True)
V3_MIN_SATISFACTION_SCORE = _env_int("MIN_SATISFACTION_SCORE", 65)
V3_MIN_PAYOFF_SCORE = _env_int("MIN_PAYOFF_SCORE", 60)
V3_MIN_CONTEXT_SCORE = _env_int("MIN_CONTEXT_SCORE", 55)
V3_MAX_CLICKBAIT_GAP = _env_int("MAX_CLICKBAIT_GAP", 45)
V3_SATISFACTION_MAX_RETRIES = _env_int("SATISFACTION_MAX_RETRIES", 2)
V3_SATISFACTION_MAX_REFINEMENT_PASSES = _env_int(
    "SATISFACTION_MAX_REFINEMENT_PASSES", 1
)
V3_SATISFACTION_FALLBACK_TOP_K = _env_int("SATISFACTION_FALLBACK_TOP_K", 3)
V3_SATISFACTION_FINAL_VALIDATION = _env_bool(
    "SATISFACTION_FINAL_VALIDATION", False
)
V3_SATISFACTION_CACHE_DIR = BASE_DIR / "cache" / "satisfaction_final"

V3_FINAL_SCORE_WEIGHTS = {
    "hook": _env_float("HOOK_WEIGHT", 0.25),
    "retention": _env_float("RETENTION_WEIGHT", 0.25),
    "satisfaction": _env_float("SATISFACTION_WEIGHT", 0.20),
    "originality": _env_float("ORIGINALITY_WEIGHT", 0.15),
    "payoff": _env_float("PAYOFF_WEIGHT", 0.10),
    "context": _env_float("CONTEXT_WEIGHT", 0.05),
}

# --------------------------------------------------
# Caption Hook Engine
# --------------------------------------------------
V3_CAPTION_HOOK_ENABLED = _env_bool("CAPTION_HOOK_ENABLED", True)
V3_CAPTION_HOOK_MIN_SCORE = _env_int("CAPTION_HOOK_MIN_SCORE", 75)
V3_CAPTION_HOOK_MAX_WORDS = _env_int("CAPTION_HOOK_MAX_WORDS", 7)
V3_CAPTION_HOOK_MAX_LINES = _env_int("CAPTION_HOOK_MAX_LINES", 2)
# One canonical duration avoids old .env min/max/default values changing the
# opening caption unexpectedly. Set CAPTION_HOOK_DURATION to override it.
V3_CAPTION_HOOK_DURATION = max(0.2, _env_float("CAPTION_HOOK_DURATION", 3.3))
V3_CAPTION_HOOK_MIN_DURATION = V3_CAPTION_HOOK_DURATION
V3_CAPTION_HOOK_MAX_DURATION = V3_CAPTION_HOOK_DURATION
V3_CAPTION_HOOK_DEFAULT_DURATION = V3_CAPTION_HOOK_DURATION
V3_CAPTION_HOOK_ALLOW_NONE = _env_bool("CAPTION_HOOK_ALLOW_NONE", True)
V3_CAPTION_HOOK_DEBUG = _env_bool("CAPTION_HOOK_DEBUG", True)
V3_CAPTION_HOOK_GENERATOR = os.getenv("CAPTION_HOOK_GENERATOR", "auto").strip().lower()
if V3_CAPTION_HOOK_GENERATOR not in {"auto", "local", "off"}:
    V3_CAPTION_HOOK_GENERATOR = "auto"
_caption_hook_prompt_base = os.getenv(
    "CAPTION_HOOK_PROMPT_VERSION",
    "caption-hook-v3-tension-nonredundancy",
).strip()
# Scoring version is appended unconditionally so an old .env prompt version
# cannot accidentally reuse cached results produced by older scoring rules.
V3_CAPTION_HOOK_PROMPT_VERSION = (
    f"{_caption_hook_prompt_base}|scoring-tension-nonredundancy-v1"
)
V3_CAPTION_HOOK_CACHE_DIR = BASE_DIR / "cache" / "caption_hooks"


def load_editorial_profile() -> dict:
    if not V3_EDITORIAL_PROFILE_PATH.exists():
        return {
            "name": V3_EDITORIAL_PROFILE_NAME,
            "editing_density": "medium",
            "sound_design_intensity": "subtle",
            "ending_behavior": "hard_cut_after_reaction",
        }
    try:
        value = json.loads(V3_EDITORIAL_PROFILE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}
