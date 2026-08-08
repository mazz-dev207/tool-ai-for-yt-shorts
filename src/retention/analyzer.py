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

RETENTION_NUM_CTX = 6144
RETENTION_NUM_PREDICT = 1400
OLLAMA_KEEP_ALIVE = "30m"


class RetentionAnalyzer:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def analyze(self, context: dict, pacing: dict, max_variants: int = 2) -> dict:
        transcript_text = format_context_for_llm(context, max_chars=11000)

        prompt = f"""
You are the retention editor for a short-form video system.
Build the strongest truthful Short from the supplied transcript without inventing anything.

RULES
- Use only facts and source ranges supported by the transcript.
- Prefer original speech/audio.
- Avoid cuts in the middle of an idea.
- Remove filler, repetition and dead time when useful.
- Prefer HOOK -> MINIMAL CONTEXT -> ESCALATION -> PAYOFF when supported.
- Keep the edit natural and understandable standalone.
- Prefer roughly 15-60 seconds.
- Use timestamps only from the transcript below.

CONTENT TYPE must be one of: {CONTENT_TYPES}.

Return ONE valid JSON object with:
1. content_type: string
2. summary: one short factual sentence
3. scores: object with integer 0-100 fields: hook, curiosity, emotion, conflict, payoff, information_density, pacing, standalone
4. retention_anchors: up to 4 objects with start, end, type, importance, reason
5. retention_risks: up to 4 objects with start, end, reason, severity
6. hook_variants: up to 3 objects with text, type, generated, score, evidence. For generated=false also include source_start/source_end.
7. open_loops: up to 3 objects with start, end, text, strength
8. pattern_interrupts: up to 3 optional visual recommendations with start, end, type, reason, importance
9. variants: up to {max_variants} EXTRACTIVE edit variants. Each has name, strategy="extractive", rationale, scores, segments.
   Each segment has start, end, role, reason.

Candidate: {context['candidate_start']:.2f}s -> {context['candidate_end']:.2f}s
Available context: {context['start']:.2f}s -> {context['end']:.2f}s

Deterministic pacing metrics:
{json.dumps(pacing, ensure_ascii=False, separators=(',', ':'))}

TIMESTAMPED TRANSCRIPT:
{transcript_text}
""".strip()

        started = time.time()
        response = ollama.chat(
            model=self.model,
            stream=False,
            think=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON. Be conservative and concise. "
                        "Ground every factual claim in the transcript."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": 0.10,
                "num_ctx": RETENTION_NUM_CTX,
                "num_predict": RETENTION_NUM_PREDICT,
            },
        )

        elapsed = time.time() - started
        load_ms = float(response.get("load_duration", 0) or 0) / 1_000_000
        prompt_tokens = int(response.get("prompt_eval_count", 0) or 0)
        output_tokens = int(response.get("eval_count", 0) or 0)

        info(
            f"Retention AI răspuns în {elapsed:.2f}s | "
            f"load={load_ms:.0f}ms | in={prompt_tokens} tok | out={output_tokens} tok"
        )

        content = response["message"]["content"].strip()
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Retention AI a returnat JSON invalid: {exc}") from exc

        if not isinstance(result, dict):
            raise RuntimeError("Retention AI nu a returnat un obiect JSON.")

        return result
