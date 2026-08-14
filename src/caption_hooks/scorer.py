from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.caption_hooks.models import CaptionHookCandidate
from src.v3_config import V3_CAPTION_HOOK_MAX_WORDS


_GENERIC_HYPE = (
    "YOU WON'T BELIEVE",
    "WATCH UNTIL THE END",
    "WHAT HAPPENS NEXT",
    "WILL SHOCK YOU",
    "CRAZIEST THING EVER",
    "THIS IS INSANE",
    "ACTUALLY INSANE",
    "OMG",
)

_CLICKBAIT = (
    "YOU NEED TO SEE",
    "MUST WATCH",
    "WATCH THIS",
    "SHOCKING",
    "UNBELIEVABLE",
)

# Common short-form AI filler. These are not automatically forbidden, but they
# should lose against a more specific/tension-driven alternative.
_OVERUSED_LANGUAGE = {
    "CHAOS": 16.0,
    "CHAOTIC": 14.0,
    "INSANE": 24.0,
    "CRAZY": 20.0,
    "UNBELIEVABLE": 35.0,
}

_STOPWORDS = {
    "THE", "A", "AN", "THIS", "THAT", "IT", "HE", "SHE", "THEY", "WE", "I",
    "IS", "ARE", "WAS", "WERE", "TO", "OF", "IN", "ON", "AT", "FOR", "AND",
    "OR", "BUT", "WITH", "BY", "FROM", "HIS", "HER", "THEIR", "OUR", "MY",
}

_TYPE_CURIOSITY = {
    "outcome_tease": 91,
    "contradiction": 95,
    "mystery": 94,
    "stakes": 94,
    "reaction_setup": 87,
    "challenge": 90,
    "specific_fact": 79,
    "conflict": 93,
    "unexpected_discovery": 91,
    "payoff_setup": 94,
    "clarification": 66,
    "none": 0,
}

# Tension is deliberately separate from curiosity. This pushes gaming captions
# toward unresolved stakes/conflict/challenge instead of flat descriptive labels.
_TYPE_TENSION = {
    "stakes": 98,
    "conflict": 96,
    "challenge": 94,
    "payoff_setup": 93,
    "contradiction": 92,
    "mystery": 90,
    "outcome_tease": 88,
    "reaction_setup": 86,
    "unexpected_discovery": 84,
    "specific_fact": 70,
    "clarification": 58,
    "none": 0,
}

_PROFILE_TYPES = {
    "gaming": {
        "outcome_tease", "contradiction", "mystery", "stakes", "reaction_setup",
        "challenge", "specific_fact", "conflict", "unexpected_discovery", "payoff_setup",
    },
    "entertainment": {
        "outcome_tease", "contradiction", "mystery", "reaction_setup",
        "challenge", "conflict", "unexpected_discovery", "payoff_setup",
    },
    "reaction": {
        "mystery", "reaction_setup", "conflict", "unexpected_discovery",
        "payoff_setup", "clarification",
    },
    "podcast": {
        "contradiction", "specific_fact", "clarification", "mystery",
    },
    "interview": {
        "contradiction", "specific_fact", "clarification", "mystery",
    },
}


def normalize_caption_text(text: str) -> str:
    value = " ".join(str(text or "").replace("\n", " ").split()).strip(" \"'.,!?")
    return value.upper()


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+(?:'[A-Z]+)?", normalize_caption_text(text))


def _content_tokens(text: str) -> list[str]:
    return [token for token in tokens(text) if token not in _STOPWORDS and len(token) > 1]


def _clamp(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 2)


def _similarity(first: str, second: str) -> float:
    a = normalize_caption_text(first)
    b = normalize_caption_text(second)
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    a_tokens = set(tokens(a))
    b_tokens = set(tokens(b))
    union = a_tokens | b_tokens
    jaccard = len(a_tokens & b_tokens) / len(union) if union else 0.0
    return max(seq, jaccard)


def split_caption_lines(text: str, max_lines: int = 2) -> list[str]:
    words = normalize_caption_text(text).split()
    if not words:
        return []
    if max_lines <= 1 or len(words) <= 3:
        return [" ".join(words)]
    best_index = min(
        range(1, len(words)),
        key=lambda index: abs(
            len(" ".join(words[:index])) - len(" ".join(words[index:]))
        ),
    )
    return [" ".join(words[:best_index]), " ".join(words[best_index:])][:max_lines]


def choose_emphasis(text: str) -> list[str]:
    words = tokens(text)
    if not words:
        return []
    for index, word in enumerate(words):
        if any(char.isdigit() for char in word):
            phrase = word
            if index + 1 < len(words) and words[index + 1] not in _STOPWORDS:
                phrase = f"{word} {words[index + 1]}"
            return [phrase]
    content = [word for word in words if word not in _STOPWORDS and len(word) >= 4]
    if not content:
        return []
    return [max(content, key=len)]


def _support_ratio(text: str, truth_context: str) -> float:
    candidate = _content_tokens(text)
    truth = set(_content_tokens(truth_context))
    if not candidate:
        return 0.0
    supported = sum(1 for token in candidate if token in truth)
    return supported / len(candidate)


def _unsupported_numbers(text: str, truth_context: str) -> list[str]:
    caption_numbers = re.findall(r"\d+(?:[.,]\d+)?", text)
    truth_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", truth_context))
    return [number for number in caption_numbers if number not in truth_numbers]


def _overused_language_penalty(text: str) -> float:
    values = tokens(text)
    return min(
        45.0,
        sum(_OVERUSED_LANGUAGE.get(token, 0.0) for token in values),
    )


def score_caption_candidate(
    candidate: CaptionHookCandidate,
    *,
    opening_text: str,
    payoff_text: str,
    truth_context: str,
    satisfaction: dict,
    content_profile: str,
) -> CaptionHookCandidate:
    candidate.text = normalize_caption_text(candidate.text)
    candidate.mode = str(candidate.mode or "COMPLEMENT").upper()
    candidate.type = str(candidate.type or "none").strip().lower()

    word_count = len(tokens(candidate.text))
    duplicate_similarity = _similarity(candidate.text, opening_text)
    payoff_similarity = _similarity(candidate.text, payoff_text)
    support_ratio = _support_ratio(candidate.text, truth_context)
    unsupported_numbers = _unsupported_numbers(candidate.text, truth_context)

    penalties = {
        "generic_hype": 0.0,
        "clickbait": 0.0,
        "spoiler": 0.0,
        "dialogue_duplication": 0.0,
        "overused_language": 0.0,
        "too_long": 0.0,
        "awkward_language": 0.0,
        "missing_context": 0.0,
        "unsupported_claim": 0.0,
    }
    rejected: list[str] = []

    if any(phrase in candidate.text for phrase in _GENERIC_HYPE):
        penalties["generic_hype"] = 45.0
        rejected.append("generic_hype")
    if any(phrase in candidate.text for phrase in _CLICKBAIT):
        penalties["clickbait"] = 30.0
        rejected.append("clickbait")

    penalties["overused_language"] = _overused_language_penalty(candidate.text)

    if word_count > V3_CAPTION_HOOK_MAX_WORDS:
        penalties["too_long"] = min(35.0, 7.0 * (word_count - V3_CAPTION_HOOK_MAX_WORDS))
    if word_count < 2:
        penalties["awkward_language"] = 18.0

    # Semantic duplication of the first spoken line is one of the fastest ways
    # for an editorial caption to feel cheap. Start penalizing earlier and make
    # near-verbatim duplication effectively noncompetitive.
    if duplicate_similarity >= 0.58:
        penalties["dialogue_duplication"] = round(
            min(68.0, 22.0 + 44.0 * duplicate_similarity),
            2,
        )
    if duplicate_similarity >= 0.76:
        rejected.append("dialogue_duplication")

    if payoff_text and payoff_similarity >= 0.72:
        if candidate.type in {"outcome_tease", "specific_fact"}:
            penalties["spoiler"] = round(8.0 + 12.0 * payoff_similarity, 2)
        else:
            penalties["spoiler"] = round(12.0 + 18.0 * payoff_similarity, 2)

    if unsupported_numbers:
        penalties["unsupported_claim"] = 100.0
        rejected.append("unsupported_number")
    elif support_ratio < 0.24 and len(_content_tokens(candidate.text)) >= 3:
        penalties["unsupported_claim"] = 28.0
        rejected.append("weak_truth_support")

    risks = satisfaction.get("risks") or {}
    if bool(risks.get("missing_context", False)) and candidate.mode != "CLARIFY":
        penalties["missing_context"] = 8.0

    clarity = 96.0
    if word_count > 7:
        clarity -= (word_count - 7) * 7.0
    if word_count < 3:
        clarity -= 8.0

    specificity = 55.0 + min(30.0, 35.0 * support_ratio)
    if any(char.isdigit() for char in candidate.text):
        specificity += 12.0
    if candidate.type in {"specific_fact", "stakes", "outcome_tease"}:
        specificity += 6.0

    naturalness = (
        92.0
        - penalties["generic_hype"] * 0.45
        - penalties["clickbait"] * 0.35
        - penalties["overused_language"] * 0.45
    )

    payoff_score = satisfaction.get("payoff_score")
    expectation_score = satisfaction.get("expectation_match_score")
    available_values = [
        float(value)
        for value in (payoff_score, expectation_score)
        if isinstance(value, (int, float))
    ]
    payoff_alignment = (
        sum(available_values) / len(available_values)
        if available_values
        else 65.0 + 25.0 * support_ratio
    )
    payoff_alignment = min(100.0, payoff_alignment + min(8.0, payoff_similarity * 10.0))

    context_score = satisfaction.get("context_independence_score")
    context_independence = (
        float(context_score)
        if isinstance(context_score, (int, float))
        else 72.0
    )
    if candidate.mode == "CLARIFY":
        context_independence = min(100.0, context_independence + 8.0)

    readability = 98.0 - penalties["too_long"] * 1.2
    line_lengths = [len(line) for line in split_caption_lines(candidate.text)]
    if line_lengths and max(line_lengths) > 25:
        readability -= min(24.0, (max(line_lengths) - 25) * 2.0)

    non_redundancy = 100.0 * (1.0 - duplicate_similarity)
    truthfulness = 100.0 - penalties["unsupported_claim"]

    curiosity = float(_TYPE_CURIOSITY.get(candidate.type, 76))
    tension = float(_TYPE_TENSION.get(candidate.type, 72))
    preferred = _PROFILE_TYPES.get(str(content_profile or "").lower())
    if preferred and candidate.type not in preferred:
        curiosity -= 8.0
        tension -= 10.0
        naturalness -= 4.0

    scores = {
        "curiosity": _clamp(curiosity),
        "tension": _clamp(tension),
        "clarity": _clamp(clarity),
        "specificity": _clamp(specificity),
        "naturalness": _clamp(naturalness),
        "payoff_alignment": _clamp(payoff_alignment),
        "context_independence": _clamp(context_independence),
        "visual_readability": _clamp(readability),
        "non_redundancy": _clamp(non_redundancy),
        "truthfulness": _clamp(truthfulness),
    }

    weights = {
        "curiosity": 0.12,
        "tension": 0.14,
        "clarity": 0.11,
        "specificity": 0.08,
        "naturalness": 0.11,
        "payoff_alignment": 0.16,
        "context_independence": 0.09,
        "visual_readability": 0.08,
        "non_redundancy": 0.06,
        "truthfulness": 0.05,
    }
    weighted = sum(scores[key] * weights[key] for key in weights)
    penalty_total = sum(penalties.values())
    final_score = _clamp(weighted - penalty_total)

    candidate.scores = scores
    candidate.penalties = {key: round(value, 2) for key, value in penalties.items()}
    candidate.score = final_score
    candidate.rejected_reasons = sorted(set(rejected))
    return candidate
