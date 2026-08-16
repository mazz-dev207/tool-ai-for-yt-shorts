from __future__ import annotations

"""Hook START semantic analyzer.

Gemini used to be the primary provider here. In the quota-optimized pipeline the
multimodal Gemini budget is reserved for the final highlight judge, so Hook START
is local-first and delegates to Qwen/Ollama.  The historical class name is kept
for import/test compatibility with the rest of the project.
"""

import json

from src.hooks.qwen_analyzer import QwenHookAnalyzer
from src.hooks.scoring import profile_priority_text
from src.logger import info


CORE_PROPERTIES = {
    "immediate_action": {"type": "integer"},
    "curiosity_gap": {"type": "integer"},
    "emotional_reaction": {"type": "integer"},
    "conflict_tension": {"type": "integer"},
    "visual_surprise": {"type": "integer"},
    "context_independence": {"type": "integer"},
    "payoff_proximity": {"type": "integer"},
}

MODIFIER_PROPERTIES = {
    "contradiction": {"type": "integer"},
    "specificity": {"type": "integer"},
    "timeframe_tension": {"type": "integer"},
    "relatability": {"type": "integer"},
    "naturalness": {"type": "integer"},
}

PENALTY_PROPERTIES = {
    "dead_air": {"type": "integer"},
    "context_dependency": {"type": "integer"},
    "spoiled_payoff": {"type": "integer"},
    "mid_sentence": {"type": "integer"},
    "duplicate_information": {"type": "integer"},
    "forced_hook": {"type": "integer"},
    "generic_setup": {"type": "integer"},
    "forced_intro": {"type": "integer"},
    "youtuber_intro": {"type": "integer"},
    "fake_hype": {"type": "integer"},
    "obvious_clickbait": {"type": "integer"},
    "repeated_context": {"type": "integer"},
    "fake_urgency": {"type": "integer"},
}

HOOK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {"type": "array"},
        "best_start": {"type": "number"},
    },
    "required": ["candidates", "best_start"],
}


def build_hook_prompt(
    *,
    original_start: float,
    highlight_end: float,
    candidate_starts: list[float],
    transcript_text: str,
    profile: str,
    context_start: float,
    context_end: float,
) -> str:
    """Compatibility/debug prompt describing the Hook START scoring contract.

    QwenHookAnalyzer owns the production local prompt; keeping this function
    preserves prompt regression tests and makes the contract inspectable.
    """
    return f"""
You are the HOOK START OPTIMIZER for YouTube Shorts / TikTok.
The highlight has ALREADY been selected. Highlight Score != Hook Score.
Choose the strongest START from the supplied candidates.

CONTENT PROFILE: {profile}
PROFILE PRIORITIES: {profile_priority_text(profile)}
ORIGINAL HIGHLIGHT: {original_start:.3f}s -> {highlight_end:.3f}s
AVAILABLE CONTEXT: {context_start:.3f}s -> {context_end:.3f}s
CANDIDATE STARTS: {json.dumps(candidate_starts, separators=(',', ':'))}

CORE SIGNALS:
- immediate_action 0-20
- curiosity_gap 0-20
- emotional_reaction 0-15
- conflict_tension 0-15
- visual_surprise 0-10
- context_independence 0-10
- payoff_proximity 0-10

HUMAN MODIFIERS:
- contradiction: -5 to +8. DO NOT penalize its absence.
- specificity: 0 to +6. DO NOT invent details.
- timeframe_tension: 0 to +5. DO NOT penalize its absence.
- relatability: 0 to +5. DO NOT penalize its absence.
- naturalness: -15 to +10.

Naturalness rewards spontaneous reaction, unfinished thought, realization,
authentic laughter/panic/confusion and specific conversational observations.
Penalize generic setup, forced intro, YouTuber intro, fake hype, clickbait,
repeated context, fake urgency, mid-sentence cuts and spoiled payoff.

FOUR-QUESTION HOOK TEST:
1. WHAT IS HAPPENING?
2. WHY SHOULD I CARE?
3. WHAT HAPPENS NEXT?
4. DOES THIS FEEL REAL?

Prefer maximum curiosity immediately BEFORE the payoff while preserving minimum
required context. Do not clip the first important word/phoneme. Local audio/frame
boundary refinement runs after this semantic choice.

TIMESTAMPED TRANSCRIPT:
{transcript_text}
""".strip()


class GeminiHookAnalyzer:
    """Backwards-compatible name for the local Qwen Hook START analyzer.

    Keeping the old symbol avoids touching optimizer/test imports while ensuring
    this stage cannot consume Gemini quota.
    """

    provider = "qwen_local"

    def __init__(self):
        self.local = QwenHookAnalyzer()
        info("[HOOK LOCAL] Hook START provider=Qwen/Ollama (Gemini reserved for Final Judge).")

    def analyze(self, **kwargs) -> tuple[dict, bool]:
        return self.local.analyze(**kwargs)
