from __future__ import annotations

import json
from pathlib import Path

from src.config import HIGHLIGHTS_DIR, OUTPUT_DIR, TRANSCRIPT_DIR
from src.v3_config import (
    V3_CONTEXT_AFTER,
    V3_CONTEXT_BEFORE,
    V3_EDITORIAL_PROFILE_NAME,
    V3_ENABLE_CONTEXT_OVERLAYS,
    V3_ENABLE_EDITORIAL_HOOKS,
    V3_ENABLE_ORIGINALITY_ENGINE,
    V3_ENABLE_ORIGINALITY_QA,
    V3_ENABLE_RECONSTRUCTED_HOOKS,
    V3_ENABLE_SEMANTIC_EFFECTS,
    V3_ENABLE_SOUND_DESIGN,
    V3_ENABLE_STORY_RESTRUCTURING,
    V3_MAX_EDIT_PLAN_RETRIES,
    V3_MAX_EFFECTS_PER_EVENT,
    V3_ORIGINALITY_MIN_SCORE,
)
from src.editing.originality_qa import run_originality_qa
from src.editing.planner import build_edit_plan
from src.highlights.gemini_judge import (
    GeminiQuotaExhausted,
    build_transcript_context,
    probe_duration,
)
from src.highlights.profiles import infer_profile
from src.hooks.v3 import select_hook_v3
from src.logger import info, success, warning
from src.originality.analyzer import analyze_originality
from src.originality.angle_generator import generate_angle
from src.originality.scoring import score_edit_plan
from src.originality.transformation_plan import choose_transformations
from src.story.restructurer import restructure_story
from src.v3.editorial_reasoner import GeminiEditorialReasoner
from src.v3.models import (
    ContextOverlay,
    EditPlan,
    HookPlan,
    OriginalityQAReport,
    OriginalityResult,
    TimelineSegment,
)


_VALID_PURPOSES = {
    "cold_open", "hook", "context", "setup", "escalation", "payoff",
    "reaction", "aftermath", "callback", "replay",
}


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _fallback_timeline(clip: dict) -> list[TimelineSegment]:
    raw = clip.get("segments") or [
        {
            "start": float(clip["start"]),
            "end": float(clip["end"]),
            "role": "context",
        }
    ]
    result = []
    for index, item in enumerate(raw):
        purpose = str(
            item.get("role")
            or ("payoff" if index == len(raw) - 1 else "context")
        ).lower()
        if purpose not in _VALID_PURPOSES:
            purpose = "context"
        result.append(
            TimelineSegment(
                source_start=float(item["start"]),
                source_end=float(item["end"]),
                purpose=purpose,
            )
        )
    return result


def _fallback_plan(
    clip_index: int,
    clip: dict,
    profile: str,
    reason: str,
) -> tuple[EditPlan, OriginalityResult, OriginalityQAReport, dict]:
    hook = HookPlan(
        mode="NATIVE",
        score=int(clip.get("hook_score", 0) or 0),
        source_start=float(clip["start"]),
        reason="V3 fallback keeps existing Hook Optimizer result.",
        confidence=float(clip.get("hook_confidence", 0.5) or 0.5),
    )
    plan = EditPlan(
        clip_index=clip_index,
        hook=hook,
        timeline=_fallback_timeline(clip),
        no_transformation_needed=True,
        metadata={
            "content_profile": profile,
            "editorial_profile": V3_EDITORIAL_PROFILE_NAME,
            "fallback_reason": reason,
        },
    )
    originality = OriginalityResult(
        originality_score=0,
        context_independence=50,
        source_dependency=50,
        weaknesses=[reason],
    )
    qa = OriginalityQAReport(
        passed=True,
        originality_score=0,
        warnings=[f"V3 fallback: {reason}"],
        checks={"fallback_to_v2": True},
    )
    return plan, originality, qa, {}


def _attach_plan_to_clip(
    clip: dict,
    plan: EditPlan,
    originality: OriginalityResult,
    qa: OriginalityQAReport,
) -> None:
    clip["segments"] = [
        {
            "start": round(item.source_start, 4),
            "end": round(item.source_end, 4),
            "role": item.purpose,
        }
        for item in plan.timeline
    ]
    if clip["segments"]:
        # For editorial timelines start/end describe the first/last played source
        # ranges. Caption and cut execution use the explicit ordered segments.
        clip["start"] = clip["segments"][0]["start"]
        clip["end"] = clip["segments"][-1]["end"]
        clip["duration"] = round(
            sum(item["end"] - item["start"] for item in clip["segments"]),
            3,
        )
    clip["v3"] = {
        "editorial_angle": plan.editorial_angle,
        "viewer_question": plan.viewer_question,
        "stakes": plan.stakes,
        "payoff": plan.payoff,
        "hook_mode": plan.hook.mode,
        "hook_score_v3": plan.hook.score,
        "originality_score": originality.originality_score,
        "qa_passed": qa.passed,
        "no_transformation_needed": plan.no_transformation_needed,
    }


def _debug_dir(video_name: str, clip_index: int) -> Path:
    path = OUTPUT_DIR / "debug" / video_name / f"clip_{clip_index}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_v3_editorial_stage(
    video_name: str,
    video_path: Path,
    requested_profile: str = "auto",
) -> Path:
    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"
    clips = _load(highlights_path)
    transcript = _load(transcript_path)

    if not isinstance(clips, list) or not clips:
        warning("[V3] Nu există highlights pentru editorial stage.")
        return highlights_path
    if not isinstance(transcript, list):
        warning("[V3] Transcript invalid; folosesc pipeline-ul V2.")
        return highlights_path
    if not V3_ENABLE_ORIGINALITY_ENGINE:
        info("[V3] Originality Engine dezactivat; pipeline-ul V2 rămâne activ.")
        return highlights_path

    duration = probe_duration(video_path)
    try:
        reasoner = GeminiEditorialReasoner()
    except Exception as exc:
        warning(
            f"[V3] Gemini editorial indisponibil: {exc}; folosesc fallback V2."
        )
        reasoner = None

    quota_exhausted = False
    for index, clip in enumerate(clips, start=1):
        profile = infer_profile(clip, requested_profile).name
        debug = _debug_dir(video_name, index)
        _save(debug / "highlight.json", clip)

        context_start = max(
            0.0,
            float(clip["start"]) - float(V3_CONTEXT_BEFORE),
        )
        context_end = min(
            duration,
            float(clip["end"]) + float(V3_CONTEXT_AFTER),
        )
        context_text = build_transcript_context(
            transcript,
            context_start,
            context_end,
        )

        if reasoner is None or quota_exhausted:
            plan, originality, qa, proposal = _fallback_plan(
                index,
                clip,
                profile,
                "gemini_editorial_unavailable",
            )
        else:
            best = None
            retry_feedback: list[str] = []
            attempts = max(1, int(V3_MAX_EDIT_PLAN_RETRIES) + 1)

            for attempt in range(attempts):
                try:
                    proposal, from_cache = reasoner.analyze(
                        video_path=video_path,
                        transcript=transcript,
                        clip=clip,
                        content_profile=profile,
                        video_duration=duration,
                        clip_index=index,
                        retry_feedback=retry_feedback,
                    )
                except GeminiQuotaExhausted as exc:
                    quota_exhausted = True
                    warning(
                        f"[V3] {exc}; restul clipurilor folosesc fallback V2."
                    )
                    plan, originality, qa, proposal = _fallback_plan(
                        index,
                        clip,
                        profile,
                        "gemini_daily_quota_exhausted",
                    )
                    best = (plan, originality, qa, proposal)
                    break
                except Exception as exc:
                    warning(
                        f"[V3] clip {index} editorial analysis failed: {exc}"
                    )
                    plan, originality, qa, proposal = _fallback_plan(
                        index,
                        clip,
                        profile,
                        f"editorial_error:{exc}",
                    )
                    best = (plan, originality, qa, proposal)
                    break

                angle = generate_angle(proposal)
                analysis = analyze_originality(
                    proposal=proposal,
                    clip=clip,
                    transcript_text=context_text,
                )
                transformations = choose_transformations(analysis, proposal)
                hook = select_hook_v3(
                    clip=clip,
                    proposal=proposal,
                    transcript=transcript,
                    context_start=context_start,
                    context_end=context_end,
                    enable_reconstructed=V3_ENABLE_RECONSTRUCTED_HOOKS,
                    enable_editorial=V3_ENABLE_EDITORIAL_HOOKS,
                )
                timeline, story_debug = restructure_story(
                    clip=clip,
                    proposal=proposal,
                    source_duration=duration,
                    enable_restructuring=(
                        V3_ENABLE_STORY_RESTRUCTURING
                        and not analysis.no_transformation_needed
                    ),
                    allowed_start=context_start,
                    allowed_end=context_end,
                )
                plan = build_edit_plan(
                    clip_index=index,
                    hook=hook,
                    angle=angle,
                    timeline=timeline,
                    proposal=proposal,
                    no_transformation_needed=analysis.no_transformation_needed,
                    content_profile=profile,
                    editorial_profile_name=V3_EDITORIAL_PROFILE_NAME,
                    enable_context_overlays=V3_ENABLE_CONTEXT_OVERLAYS,
                    enable_semantic_effects=V3_ENABLE_SEMANTIC_EFFECTS,
                    enable_sound_design=V3_ENABLE_SOUND_DESIGN,
                    max_effects_per_event=V3_MAX_EFFECTS_PER_EVENT,
                )
                plan.metadata.update(
                    {
                        "story": story_debug,
                        "transformations": transformations,
                        "proposal_cache": from_cache,
                        "analyzed_context": [
                            round(context_start, 3),
                            round(context_end, 3),
                        ],
                    }
                )

                if (
                    hook.mode == "EDITORIAL"
                    and hook.text
                    and V3_ENABLE_CONTEXT_OVERLAYS
                    and not any(item.text == hook.text for item in plan.context_overlays)
                ):
                    plan.context_overlays.insert(
                        0,
                        ContextOverlay(
                            text=hook.text,
                            start=0.05,
                            duration=hook.duration or 1.2,
                            purpose="hook",
                        ),
                    )

                originality = score_edit_plan(plan, analysis)
                qa = run_originality_qa(
                    plan=plan,
                    originality=originality,
                    min_score=(
                        V3_ORIGINALITY_MIN_SCORE
                        if V3_ENABLE_ORIGINALITY_QA
                        else 0
                    ),
                    source_duration=duration,
                    max_effects_per_event=V3_MAX_EFFECTS_PER_EVENT,
                )
                candidate = (plan, originality, qa, proposal)
                if (
                    best is None
                    or originality.originality_score
                    > best[1].originality_score
                ):
                    best = candidate

                info(
                    f"[ANGLE] clip={index} {angle.primary_angle} | "
                    f"[STORY] segments={len(plan.timeline)} "
                    f"reordered={story_debug.get('reordered', False)}"
                )
                info(
                    f"[ORIGINALITY] clip={index} attempt={attempt + 1} "
                    f"score={originality.originality_score} "
                    f"qa={'pass' if qa.passed else 'retry'}"
                )
                if qa.passed:
                    break

                retry_feedback = (
                    originality.weaknesses + qa.problems
                )[:8]
                if attempt + 1 < attempts:
                    info(
                        "[ORIGINALITY] regenerating edit plan "
                        f"weaknesses={retry_feedback}"
                    )

            if best is None:
                plan, originality, qa, proposal = _fallback_plan(
                    index,
                    clip,
                    profile,
                    "no_valid_v3_plan",
                )
            else:
                plan, originality, qa, proposal = best
                if not qa.passed:
                    warning(
                        f"[ORIGINALITY] clip={index} below target after retries; "
                        f"using best valid plan score={originality.originality_score}"
                    )

        _attach_plan_to_clip(clip, plan, originality, qa)

        plan_dir = HIGHLIGHTS_DIR / "v3" / video_name / f"clip_{index}"
        _save(plan_dir / "edit_plan.json", plan.to_dict())

        _save(debug / "hook_analysis.json", plan.hook.__dict__)
        _save(
            debug / "editorial_angle.json",
            {
                "primary_angle": plan.editorial_angle,
                "viewer_question": plan.viewer_question,
                "stakes": plan.stakes,
                "payoff": plan.payoff,
            },
        )
        _save(
            debug / "originality_analysis.json",
            proposal.get("originality_analysis", {}) if proposal else {},
        )
        _save(debug / "edit_plan.json", plan.to_dict())
        _save(debug / "originality_qa.json", qa.__dict__)

        info(
            f"[HOOK] clip={index} selected={plan.hook.mode} "
            f"score={plan.hook.score} | "
            f"[EDIT] visual_events={len(plan.visual_events)} "
            f"audio_events={len(plan.audio_events)}"
        )

    _save(highlights_path, clips)
    success(f"[V3] Editorial stage terminat pentru {len(clips)} Shorts.")
    return highlights_path
