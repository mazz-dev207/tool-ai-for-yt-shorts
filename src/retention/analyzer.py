from __future__ import annotations

import json
import time

import ollama

from src.config import OLLAMA_MODEL
from src.logger import info
from src.retention.context import format_context_for_llm


CONTENT_TYPES = (
    "podcast, interview, gaming, challenge, entertainment, livestream, commentary, "
    "documentary, educational, reaction, storytelling, general"
)


class RetentionAnalyzer:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def analyze(self, context: dict, pacing: dict, max_variants: int = 3) -> dict:
        transcript_text = format_context_for_llm(context)

        prompt = f"""
You are the retention editor for a short-form video system.
Your job is NOT to invent a story. Your job is to build the strongest truthful Short possible from the supplied transcript.

NON-NEGOTIABLE RULES:
- Never invent people, money, numbers, quotes, events, consequences, stakes or context.
- Every factual claim in a hook must be supported by the transcript.
- Extractive variants may ONLY use source timestamp ranges that exist in the supplied transcript.
- Prefer original speech/audio.
- Avoid cuts in the middle of an idea.
- Avoid robotic edits and unnatural sentence combinations.
- The final story should progress: HOOK -> MINIMAL CONTEXT -> ESCALATION -> PAYOFF when the material supports it.
- Do not force a fixed duration. Prefer roughly 15-60 seconds depending on the idea.
- Generated hooks are suggestions only; this project currently has no TTS, so extractive hooks are preferred for the selected audio edit.
- Use timestamps only from the transcript below.
- Adapt priorities to content type: podcast/interview -> strong statements, stories, revelations; gaming -> clutch/fail/win/rare events/reactions; challenge/entertainment -> stakes, progress, eliminations, twists, results; educational -> surprising facts, clear explanations, myths, consequences; storytelling -> conflict, mystery, escalation, twist and payoff.

CONTENT TYPE must be one of: {CONTENT_TYPES}.

Return ONE valid JSON object with these keys:
1. content_type: string
2. summary: short factual summary
3. scores: object with integer 0-100 fields: hook, curiosity, emotion, conflict, payoff, information_density, pacing, standalone
4. retention_anchors: array of objects with start, end, type, importance (0-100), reason
5. retention_risks: array of objects with start, end, reason, severity (0-100)
6. hook_variants: 3-5 objects. Each has text, type, generated, score, evidence. If generated=false also include source_start and source_end.
7. open_loops: array of truthful open loops already present or naturally implied by the material. Each has start, end, text, strength (0-100). Do not invent artificial promises.
8. pattern_interrupts: array of OPTIONAL visual recommendations. Each has start, end, type (crop_change/zoom/punch_in/speaker_change/broll/text/effect/caption_position), reason, importance (0-100). Only recommend when attention would benefit.
9. variants: up to {max_variants} EXTRACTIVE edit variants. Each object has name, strategy="extractive", rationale, scores (same 8 score fields), and segments.
   Each segment has start, end, role (hook/context/escalation/payoff/bridge/reaction), reason.
   Segments may skip filler, pauses, repetition and tangents. Reordering is allowed only when it sounds natural and preserves meaning.

The candidate starts at {context['candidate_start']:.2f}s and ends at {context['candidate_end']:.2f}s.
The available context starts at {context['start']:.2f}s and ends at {context['end']:.2f}s.

Deterministic pacing metrics for the original candidate:
{json.dumps(pacing, ensure_ascii=False)}

TIMESTAMPED TRANSCRIPT:
{transcript_text}
""".strip()

        started = time.time()
        response = ollama.chat(
            model=self.model,
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON. Be conservative. Ground every factual claim in the supplied transcript. "
                        "Do not default all scores to the same value."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.15, "think": False},
        )

        elapsed = time.time() - started
        info(f"Retention AI răspuns în {elapsed:.2f}s")

        content = response["message"]["content"].strip()
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Retention AI a returnat JSON invalid: {exc}") from exc

        if not isinstance(result, dict):
            raise RuntimeError("Retention AI nu a returnat un obiect JSON.")

        return result
