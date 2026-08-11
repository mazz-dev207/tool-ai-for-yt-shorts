from __future__ import annotations

import json
from pathlib import Path

from src.config import HIGHLIGHTS_DIR, OUTPUT_DIR
from src.logger import success, warning
from src.strategy.models import ExperimentMetadata


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_experiment_metadata(video_name: str) -> Path | None:
    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    if not highlights_path.exists():
        warning("[EXPERIMENT] Highlights missing; metadata export skipped.")
        return None

    clips = _load(highlights_path)
    if not isinstance(clips, list):
        warning("[EXPERIMENT] Highlights JSON invalid; metadata export skipped.")
        return None

    items = []
    for index, clip in enumerate(clips, start=1):
        v3 = clip.get("v3") or {}
        metadata = ExperimentMetadata(
            short_id=f"{video_name}_clip_{index}",
            format_id=str(clip.get("format_id", "unclassified") or "unclassified"),
            format_version=int(clip.get("format_version", 1) or 1),
            content_opportunity_score=float(clip.get("content_opportunity_score", 0) or 0),
            hook_type=str(v3.get("hook_mode") or clip.get("hook_type") or "NATIVE"),
            hook_score=float(v3.get("hook_score_v3", clip.get("hook_score", 0)) or 0),
            originality_score=float(v3.get("originality_score", 0) or 0),
            duration=float(clip.get("duration", 0) or 0),
        )
        item = metadata.to_dict()
        item["format_fit_score"] = float(clip.get("format_fit_score", 0) or 0)
        item["strategic_gate"] = str(clip.get("strategic_gate", "UNKNOWN"))
        items.append(item)

        debug_dir = OUTPUT_DIR / "debug" / video_name / f"clip_{index}"
        _save(debug_dir / "experiment_metadata.json", item)

    output = OUTPUT_DIR / "debug" / video_name / "experiment_metadata.json"
    _save(
        output,
        {
            "video": video_name,
            "shorts": items,
            "note": "Performance fields are intentionally empty until imported from YouTube/TikTok analytics.",
        },
    )
    success(f"[EXPERIMENT] Exported metadata for {len(items)} Shorts: {output}")
    return output
