from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import HIGHLIGHTS_DIR, OUTPUT_DIR
from src.logger import success, warning
from src.strategy.models import ExperimentMetadata


PERFORMANCE_FIELDS = (
    "views",
    "viewed_vs_swiped",
    "average_view_duration",
    "average_percentage_viewed",
    "likes",
    "comments",
    "shares",
    "subscribers_gained",
)


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
        strategy_context = clip.get("strategy_context") or {}
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
        if strategy_context:
            _save(debug_dir / "format_profile.json", strategy_context.get("format_profile", {}))
            _save(debug_dir / "format_match.json", strategy_context.get("format_match", {}))
            _save(debug_dir / "content_opportunity.json", strategy_context.get("content_opportunity", {}))
            _save(debug_dir / "channel_strategy.json", strategy_context.get("channel_strategy", {}))

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


def _normalized_performance(raw: dict) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {key: None for key in PERFORMANCE_FIELDS}
    if not isinstance(raw, dict):
        return result
    integer_fields = {"views", "likes", "comments", "shares", "subscribers_gained"}
    for key in PERFORMANCE_FIELDS:
        if key not in raw or raw.get(key) is None:
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        result[key] = int(round(value)) if key in integer_fields else round(value, 4)
    return result


def import_performance_metadata(video_name: str, performance_path: Path) -> Path:
    """Associate manually exported analytics with existing V3 experiment metadata.

    Accepted input is either a list of objects or {"shorts": [...]}. Each item must
    contain short_id and may contain the supported performance fields. This does not
    fetch platform analytics and does not change ChannelStrategy automatically.
    """
    aggregate_path = OUTPUT_DIR / "debug" / video_name / "experiment_metadata.json"
    if not aggregate_path.exists():
        generated = export_experiment_metadata(video_name)
        if generated is None or not aggregate_path.exists():
            raise FileNotFoundError(
                f"Experiment metadata missing for {video_name}; run V3 first."
            )

    raw_import = _load(performance_path)
    imported_items = raw_import.get("shorts", []) if isinstance(raw_import, dict) else raw_import
    if not isinstance(imported_items, list):
        raise ValueError("Performance import must be a list or an object with a shorts list")

    by_id: dict[str, dict] = {}
    for raw in imported_items:
        if not isinstance(raw, dict):
            continue
        short_id = str(raw.get("short_id", "") or "").strip()
        if not short_id:
            continue
        performance = raw.get("performance") if isinstance(raw.get("performance"), dict) else raw
        by_id[short_id] = _normalized_performance(performance)

    aggregate = _load(aggregate_path)
    shorts = aggregate.get("shorts", []) if isinstance(aggregate, dict) else []
    if not isinstance(shorts, list):
        raise ValueError("Existing experiment metadata has an invalid shorts field")

    updated = 0
    for index, item in enumerate(shorts, start=1):
        if not isinstance(item, dict):
            continue
        short_id = str(item.get("short_id", "") or "")
        incoming = by_id.get(short_id)
        if incoming is None:
            continue
        existing = item.get("performance") if isinstance(item.get("performance"), dict) else {}
        merged = {key: existing.get(key) for key in PERFORMANCE_FIELDS}
        for key, value in incoming.items():
            if value is not None:
                merged[key] = value
        item["performance"] = merged
        _save(
            OUTPUT_DIR / "debug" / video_name / f"clip_{index}" / "experiment_metadata.json",
            item,
        )
        updated += 1

    aggregate["shorts"] = shorts
    aggregate["performance_import"] = {
        "source_file": str(performance_path),
        "matched_shorts": updated,
        "submitted_shorts": len(by_id),
    }
    _save(aggregate_path, aggregate)
    success(
        f"[EXPERIMENT] Imported performance for {updated}/{len(by_id)} Shorts: {aggregate_path}"
    )
    return aggregate_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="AI Shorts V3 experiment metadata export/import"
    )
    parser.add_argument("video_name")
    parser.add_argument(
        "--import-performance",
        dest="performance_path",
        default=None,
        help="JSON with short_id + performance metrics to associate with rendered Shorts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.performance_path:
        import_performance_metadata(
            args.video_name,
            Path(args.performance_path).expanduser().resolve(),
        )
    else:
        export_experiment_metadata(args.video_name)