import json
import sys
from pathlib import Path

from src.config import (
    TRANSCRIPT_DIR,
    HIGHLIGHTS_DIR,
    SUBTITLES_DIR,
    CAPTION_ANIMATION,
)
from src.logger import info, success, warning
from src.caption.styles import DEFAULT_STYLE
from src.caption.animations import Animation
from src.caption.grouping import group_words, WordGroup, Word
from src.caption.ass_writer import ASSWriter
from src.caption_hooks.generator import generate_caption_hook
from src.caption_hooks.models import CaptionHookResult
from src.caption_hooks.renderer import append_caption_hook_to_ass
from src.retention.timeline import extract_remapped_words


class CaptionEngine:
    def __init__(self):
        self.style = DEFAULT_STYLE
        self.animation = getattr(Animation, CAPTION_ANIMATION.upper())
        self.writer = ASSWriter(style=self.style, animation=self.animation)

    def load_json(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def save_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_transcript(self, video_name: str):
        return self.load_json(TRANSCRIPT_DIR / f"{video_name}.json")

    def load_highlights(self, video_name: str):
        return self.load_json(HIGHLIGHTS_DIR / f"{video_name}.json")

    def edit_plan_path(self, video_name: str, index: int) -> Path:
        return HIGHLIGHTS_DIR / "v3" / video_name / f"clip_{index}" / "edit_plan.json"

    def load_edit_plan(self, video_name: str, index: int) -> dict | None:
        path = self.edit_plan_path(video_name, index)
        if not path.exists():
            return None
        try:
            value = self.load_json(path)
            return value if isinstance(value, dict) else None
        except Exception as exc:
            warning(f"[CAPTION HOOK] clip={index} edit_plan unreadable: {exc}")
            return None

    def extract_words(self, transcript, clip_start, clip_end):
        words = []
        for segment in transcript:
            if segment["end"] < clip_start or segment["start"] > clip_end:
                continue

            for word in segment.get("words", []):
                if word["end"] < clip_start or word["start"] > clip_end:
                    continue

                words.append(
                    {
                        "word": word["word"],
                        "start": float(word["start"]) - clip_start,
                        "end": float(word["end"]) - clip_start,
                    }
                )
        return words

    def validate_groups(self, groups):
        fixed = []
        for group in groups:
            if isinstance(group, WordGroup):
                fixed.append(group)
                continue

            if isinstance(group, list):
                words = []
                for item in group:
                    if isinstance(item, Word):
                        words.append(item)
                    elif isinstance(item, dict):
                        words.append(
                            Word(
                                word=item["word"],
                                start=float(item["start"]),
                                end=float(item["end"]),
                            )
                        )
                if words:
                    fixed.append(
                        WordGroup(words=words, start=words[0].start, end=words[-1].end)
                    )
        return fixed

    def _caption_hook(
        self,
        *,
        video_name: str,
        transcript: list[dict],
        clip: dict,
        index: int,
        output: Path,
    ) -> CaptionHookResult:
        edit_plan = self.load_edit_plan(video_name, index)
        if edit_plan is None:
            # Keep the V2 execution path backward compatible. Caption Hook is a
            # V3 presentation layer and is not injected into legacy-only runs.
            return CaptionHookResult(
                enabled=False,
                status="disabled",
                reason="No V3 edit_plan available; legacy captions remain unchanged.",
            )

        metadata = edit_plan.get("metadata") or {}
        profile = str(
            metadata.get("content_profile")
            or (clip.get("v3") or {}).get("content_profile")
            or clip.get("discovery_profile")
            or "general"
        ).strip().lower()

        try:
            result = generate_caption_hook(
                video_name=video_name,
                clip_index=index,
                clip=clip,
                transcript=transcript,
                edit_plan=edit_plan,
                content_profile=profile,
            )
        except Exception as exc:
            warning(
                f"[CAPTION HOOK] clip={index} generation failed: {exc}; "
                "continuing with normal subtitles."
            )
            result = CaptionHookResult(
                enabled=False,
                status="unavailable",
                reason=f"Caption Hook generation failed safely: {exc}",
            )

        payload = result.to_dict()
        clip["caption_hook"] = payload
        edit_plan["caption_hook"] = payload
        try:
            self.save_json(self.edit_plan_path(video_name, index), edit_plan)
        except Exception as exc:
            warning(f"[CAPTION HOOK] clip={index} could not persist edit_plan result: {exc}")

        if result.enabled:
            rendered = append_caption_hook_to_ass(output, result)
            if not rendered:
                # The selection remains useful for debug, but the media pipeline
                # must not pretend the overlay was burned if ASS injection failed.
                clip["caption_hook"]["render_status"] = "failed"
                edit_plan["caption_hook"]["render_status"] = "failed"
            else:
                clip["caption_hook"]["render_status"] = "ready"
                edit_plan["caption_hook"]["render_status"] = "ready"
            try:
                self.save_json(self.edit_plan_path(video_name, index), edit_plan)
            except Exception:
                pass

        return result

    def generate_clip(self, video_name, transcript, clip, index):
        info(f"Generez subtitrarea {index}")

        if clip.get("segments"):
            words = extract_remapped_words(transcript, clip["segments"])
            info(
                f"Clip {index}: timestamp-urile au fost remapate pentru "
                f"{len(clip['segments'])} segmente extractive."
            )
        else:
            words = self.extract_words(
                transcript,
                float(clip["start"]),
                float(clip["end"]),
            )

        if not words:
            info(f"Clip {index} nu are cuvinte.")
            return

        groups = self.validate_groups(group_words(words))
        info(f"Au fost create {len(groups)} grupuri.")

        if not groups:
            info(f"Clip {index} fără grupuri.")
            return

        output = SUBTITLES_DIR / f"clip_{index}.ass"
        self.writer.write(groups, output)
        success(f"Creat {output.name}")

        self._caption_hook(
            video_name=video_name,
            transcript=transcript,
            clip=clip,
            index=index,
            output=output,
        )

    def generate(self, video_name: str):
        transcript = self.load_transcript(video_name)
        highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
        highlights = self.load_highlights(video_name)
        SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)

        generated = 0
        for index, clip in enumerate(highlights, start=1):
            self.generate_clip(video_name, transcript, clip, index)
            generated += 1

        # Persist Caption Hook structured output next to Satisfaction/V3 metadata.
        # This mutates only the presentation metadata; highlight boundaries/order
        # remain exactly as selected upstream.
        self.save_json(highlights_path, highlights)

        success(f"Generate {generated} fișiere ASS.")
        return SUBTITLES_DIR


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Utilizare: python src/caption_engine.py "video_name"')
        sys.exit(1)

    try:
        CaptionEngine().generate(" ".join(sys.argv[1:]))
    except Exception as exc:
        print(f"Eroare: {exc}")
        sys.exit(1)
