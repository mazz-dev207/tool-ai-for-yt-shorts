from pathlib import Path
import json
import os

from src.config import BASE_DIR


def _env_bool(name: str, default: bool) -> bool:
    raw=os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1","true","yes","on"}


def _env_int(name: str, default: int) -> int:
    try:return int(os.getenv(name,str(default)))
    except ValueError:return default


def _env_float(name: str, default: float) -> float:
    try:return float(os.getenv(name,str(default)))
    except ValueError:return default


V3_ENABLED=_env_bool("V3_ENABLED",True)
V3_ENABLE_ORIGINALITY_ENGINE=_env_bool("ENABLE_ORIGINALITY_ENGINE",True)
V3_ENABLE_STORY_RESTRUCTURING=_env_bool("ENABLE_STORY_RESTRUCTURING",True)
V3_ENABLE_EDITORIAL_HOOKS=_env_bool("ENABLE_EDITORIAL_HOOKS",True)
V3_ENABLE_RECONSTRUCTED_HOOKS=_env_bool("ENABLE_RECONSTRUCTED_HOOKS",True)
V3_ENABLE_CONTEXT_OVERLAYS=_env_bool("ENABLE_CONTEXT_OVERLAYS",True)
V3_ENABLE_SEMANTIC_EFFECTS=_env_bool("ENABLE_SEMANTIC_EFFECTS",True)
V3_ENABLE_SOUND_DESIGN=_env_bool("ENABLE_SOUND_DESIGN",True)
V3_ENABLE_ORIGINALITY_QA=_env_bool("ENABLE_ORIGINALITY_QA",True)

V3_ORIGINALITY_MIN_SCORE=_env_int("ORIGINALITY_MIN_SCORE",60)
V3_MAX_EDIT_PLAN_RETRIES=_env_int("MAX_EDIT_PLAN_RETRIES",2)
V3_MAX_EFFECTS_PER_EVENT=_env_int("MAX_EFFECTS_PER_EVENT",2)
V3_CONTEXT_BEFORE=_env_float("V3_CONTEXT_BEFORE",8.0)
V3_CONTEXT_AFTER=_env_float("V3_CONTEXT_AFTER",8.0)
V3_EDITORIAL_PROMPT_VERSION=os.getenv("V3_EDITORIAL_PROMPT_VERSION","ai-shorts-v3-editorial-1").strip()
V3_CACHE_DIR=BASE_DIR/"cache"/"v3_editorial"
V3_SFX_DIR=BASE_DIR/"assets"/"sfx"
V3_EDITORIAL_PROFILE_NAME=os.getenv("V3_EDITORIAL_PROFILE","mazclips").strip().lower()
V3_EDITORIAL_PROFILE_PATH=BASE_DIR/"config"/"editorial_profile.json"


def load_editorial_profile() -> dict:
    if not V3_EDITORIAL_PROFILE_PATH.exists():
        return {
            "name":V3_EDITORIAL_PROFILE_NAME,
            "editing_density":"medium",
            "sound_design_intensity":"subtle",
            "ending_behavior":"hard_cut_after_reaction",
        }
    try:
        value=json.loads(V3_EDITORIAL_PROFILE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value,dict) else {}
    except Exception:
        return {}
