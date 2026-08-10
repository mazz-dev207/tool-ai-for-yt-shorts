from __future__ import annotations

import json
import re
import time

import ollama

from src.config import OLLAMA_MODEL
from src.logger import info, warning
from src.retention.context import format_context_for_llm


CONTENT_TYPES = (
    "podcast, interview, gaming, challenge, entertainment, livestream, commentary, "
    "documentary, educational, reaction, storytelling, general"
)

RETENTION_NUM_CTX = 4096
RETENTION_NUM_PREDICT = 800
RETENTION_RETRY_NUM_PREDICT = 600
OLLAMA_KEEP_ALIVE = "30m"


def parse_retention_json(content: str) -> dict:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])

    last_error: Exception | None = None
    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        attempts = [
            candidate,
            re.sub(r",\s*([}\]])", r"\1", candidate),
        ]
        for attempt in attempts:
            try:
                result = json.loads(attempt)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as exc:
                last_error = exc

            try:
                result, _end = decoder.raw_decode(attempt.lstrip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as exc:
                last_error = exc

    raise RuntimeError(f"Retention AI a returnat JSON invalid: {last_error}")


def _with_optimizer_defaults(result: dict) -> dict:
    if not isinstance(result, dict):
        raise RuntimeError("Retention AI nu a returnat un obiect JSON.")

    result.setdefault("hook_variants", [])
    result.setdefault("variants", [])
    result.setdefault("scores", {})
    result.setdefault("retention_anchors", [])
    result.setdefault("retention_risks", [])
    result.setdefault("open_loops", [])
    result.setdefault("pattern_interrupts", [])
    return result


class RetentionAnalyzer:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def _chat(
        self,
        *,
        prompt: str,
        system_prompt: str,
        num_predict: int,
        temperature: float,
        label: str,
    ) -> tuple[str, int]:
        started = time.time()
        response = ollama.chat(
            model=self.model,
            stream=False,
            think=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": temperature,
                "num_ctx": RETENTION_NUM_CTX,
                "num_predict": num_predict,
            },
        )

        elapsed = time.time() - started
        load_ms = float(response.get("load_duration", 0) or 0) / 1_000_000
        prompt_tokens = int(response.get("prompt_eval_count", 0) or 0)
        output_tokens = int(response.get("eval_count", 0) or 0)

        info(
            f"Retention AI {label} în {elapsed:.2f}s | "
            f"load={load_ms:.0f}ms | in={prompt_tokens} tok | out={output_tokens} tok"
        )
        if output_tokens >= num_predict - 5:
            warning(
                f"[RETENTION] {label} a atins limita de output ({num_predict}); "
                "răspunsul poate fi trunchiat."
            )

        return str(response["message"]["content"]), output_tokens

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
- Output must be COMPACT. Target under 450 tokens.
- Use at most 4 edit segments.
- Segment reason must be at most 4 words.
- Do not repeat the top-level scores inside the edit variant.
- Do not explain your reasoning outside JSON.

CONTENT TYPE must be one of: {CONTENT_TYPES}.

Return ONLY this compact JSON structure:
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
      "text": "short source-supported hook",
      "type": "original",
      "generated": false,
      "score": 0,
      "source_start": 0.0,
      "source_end": 0.0
    }}
  ],
  "variants": [
    {{
      "name": "best_edit",
      "strategy": "extractive",
      "segments": [
        {{"start": 0.0, "end": 0.0, "role": "hook", "reason": "short reason"}}
      ]
    }}
  ]
}}

Return at most ONE hook and ONE edit variant.
Do not add rationale, duplicate scores, evidence, summaries, anchors, risks, open loops,
visual suggestions, markdown, comments, or extra keys.

Candidate: {context['candidate_start']:.2f}s -> {context['candidate_end']:.2f}s
Available context: {context['start']:.2f}s -> {context['end']:.2f}s

Deterministic pacing metrics:
{json.dumps(pacing, ensure_ascii=False, separators=(',', ':'))}

TIMESTAMPED TRANSCRIPT:
{transcript_text}
""".strip()

        content, output_tokens = self._chat(
            prompt=prompt,
            system_prompt=(
                "Return only one compact valid JSON object. Stay under 450 tokens. "
                "Use at most four extractive segments and ground every timestamp in the transcript."
            ),
            num_predict=RETENTION_NUM_PREDICT,
            temperature=0.05,
            label="răspuns",
        )

        try:
            return _with_optimizer_defaults(parse_retention_json(content))
        except RuntimeError as first_error:
            warning(
                f"[RETENTION] Primul JSON este invalid/trunchiat: {first_error}. "
                "Execut un singur retry compact."
            )

        retry_prompt = f"""
The previous response was invalid or truncated.
Return ONLY a complete ultra-compact JSON object. Maximum 300 tokens.
No prose. No rationale. No evidence. No duplicate scores. At most 4 segments.
If no original hook can be represented safely, use "hook_variants": [].

Required shape:
{{
  "content_type":"general",
  "scores":{{"hook":0,"curiosity":0,"emotion":0,"conflict":0,"payoff":0,"information_density":0,"pacing":0,"standalone":0}},
  "hook_variants":[],
  "variants":[{{"name":"best_edit","strategy":"extractive","segments":[{{"start":0.0,"end":0.0,"role":"hook"}}]}}]
}}

Candidate: {context['candidate_start']:.2f}s -> {context['candidate_end']:.2f}s
Available context: {context['start']:.2f}s -> {context['end']:.2f}s
TIMESTAMPED TRANSCRIPT:
{transcript_text}
""".strip()

        retry_content, _retry_tokens = self._chat(
            prompt=retry_prompt,
            system_prompt=(
                "This is a JSON repair retry. Return one COMPLETE compact JSON object only, "
                "under 300 tokens. Never continue after the closing brace."
            ),
            num_predict=RETENTION_RETRY_NUM_PREDICT,
            temperature=0.0,
            label="retry",
        )

        try:
            result = parse_retention_json(retry_content)
        except RuntimeError as retry_error:
            raise RuntimeError(
                f"Retention AI JSON invalid și după retry compact: {retry_error}"
            ) from retry_error

        return _with_optimizer_defaults(result)
