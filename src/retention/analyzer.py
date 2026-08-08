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

# Retention-ul trebuie să fie suficient de detaliat pentru editare, nu pentru
# a produce metadata pe care pipeline-ul nu o folosește la tăiere.
RETENTION_NUM_CTX = 4096
RETENTION_NUM_PREDICT = 900
OLLAMA_KEEP_ALIVE = "30m"


class RetentionAnalyzer:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def analyze(self, context: dict, pacing: dict, max_variants: int = 1) -> dict:
        transcript_text = format_context_for_llm(context, max_chars=8500)

        prompt = f"""
You are a retention editor for short-form video.
Create the strongest truthful EXTRACTIVE edit from the transcript.

RULES
- Never invent facts, dialogue, stakes or events.
- Use only timestamp ranges present in the transcript.
- Prefer original speech/audio and natural sentence boundaries.
- Remove filler, repetition and dead time only when the edit still sounds natural.
- Prefer HOOK -> MINIMAL CONTEXT -> ESCALATION -> PAYOFF when supported.
- Keep the result understandable without the original video.
- Target roughly 15-60 seconds, but quality matters more than duration.
- Be concise. The complete JSON response should normally stay under 650 tokens.

CONTENT TYPE must be one of: {CONTENT_TYPES}.

Return ONLY one valid JSON object with exactly these keys:
{{
  "content_type": "...",
  "scores": {{
    "hook": 0,
    "curiosity": 0,
    "emotion": 0,
    "conflict": 0,
    "payoff": 0,
    "information_density": 0,
    "pacing": 0,
    "standalone": 0
  }},
  "hook_variants": [
    {{
      "text": "short exact/source-supported hook description",
      "type": "original",
      "generated": false,
      "score": 0,
      "evidence": [],
      "source_start": 0.0,
      "source_end": 0.0
    }}
  ],
  "variants": [
    {{
      "name": "best_edit",
      "strategy": "extractive",
      "rationale": "one short sentence",
      "scores": {{
        "hook": 0,
        "curiosity": 0,
        "emotion": 0,
        "conflict": 0,
        "payoff": 0,
        "information_density": 0,
        "pacing": 0,
        "standalone": 0
      }},
      "segments": [
        {{"start": 0.0, "end": 0.0, "role": "hook", "reason": "short reason"}}
      ]
    }}
  ]
}}

Return at most ONE hook and ONE edit variant.
Use no more than 6 segments in the edit.
Do not add summaries, anchors, risks, open loops, visual suggestions, explanations, markdown or extra keys.

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
                        "Return only compact valid JSON matching the requested structure. "
                        "Ground every timestamp and factual claim in the transcript."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": 0.08,
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
            raise RuntimeError(
                f"Retention AI a returnat JSON invalid: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError("Retention AI nu a returnat un obiect JSON.")

        # Câmpurile următoare sunt opționale pentru optimizer. Le păstrăm
        # compatibile fără să cerem modelului să consume tokeni pentru ele.
        result.setdefault("retention_anchors", [])
        result.setdefault("retention_risks", [])
        result.setdefault("open_loops", [])
        result.setdefault("pattern_interrupts", [])

        return result
