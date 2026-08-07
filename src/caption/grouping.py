from dataclasses import dataclass
from typing import List
import re

from src.caption.utils import (
    clean_word,
    has_pause,
    has_sentence_end,
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MAX_WORDS = 3
MIN_WORDS = 1

# Limite absolute. Ritmul vorbirii va alege dinamic
# o durată-țintă în interiorul acestor limite.
MIN_DISPLAY_DURATION = 0.38
MAX_DURATION = 1.85

# Pauzele mari sunt tratate ca limite naturale de idee.
HARD_PAUSE_THRESHOLD = 0.55
PAUSE_THRESHOLD = 0.35
SOFT_PAUSE_THRESHOLD = 0.18

# Praguri pentru viteza locală a vorbirii (words/sec).
SLOW_WPS = 1.75
FAST_WPS = 3.10

LOCAL_RATE_RADIUS = 3


# --------------------------------------------------
# RULE-BASED LANGUAGE KNOWLEDGE
# --------------------------------------------------

COMMON_STARTERS = {
    "și", "iar", "dar", "că", "ca", "de", "la", "în", "pe", "cu",
    "din", "spre", "pentru", "să", "sa", "se", "ori",
    "to", "of", "the", "a", "an", "and", "or", "but", "for",
    "in", "on", "at", "with",
}

COMMON_ENDERS = {
    "și", "iar", "dar", "că", "ca", "de", "la", "în", "pe", "cu",
    "din", "spre", "pentru", "să", "sa",
    "to", "of", "the", "a", "an", "and", "or", "but", "for",
    "in", "on", "at", "with",
}

# Expresiile pot avea 2, 3 sau 4 cuvinte. Dacă o posibilă
# despărțire taie una dintre ele, algoritmul o penalizează puternic.
COMMON_PHRASES = {
    ("nu", "știu"),
    ("nu", "vreau"),
    ("nu", "pot"),
    ("nu", "mai"),
    ("nu", "cred"),
    ("nu", "este"),
    ("nu", "e"),

    ("ce", "faci"),
    ("ce", "este"),
    ("ce", "spui"),
    ("ce", "înseamnă"),

    ("hai", "să"),
    ("trebuie", "să"),
    ("vreau", "să"),
    ("poți", "să"),
    ("pot", "să"),

    ("de", "fapt"),
    ("de", "exemplu"),
    ("de", "aceea"),
    ("de", "ce"),
    ("în", "general"),
    ("în", "schimb"),
    ("în", "primul", "rând"),
    ("în", "cele", "din", "urmă"),

    ("o", "să"),
    ("am", "fost"),
    ("a", "fost"),
    ("au", "fost"),

    ("you", "know"),
    ("i", "think"),
    ("i", "know"),
    ("i", "want"),
    ("have", "to"),
    ("going", "to"),
    ("kind", "of"),
    ("sort", "of"),
    ("thank", "you"),
}

NUMBER_UNITS = {
    "secundă", "secunde", "minute", "minut", "oră", "ore",
    "zi", "zile", "săptămână", "săptămâni", "lună", "luni",
    "an", "ani", "lei", "euro", "dolari", "%", "procente",
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
}

CLAUSE_ENDINGS = (",", ";", ":")


# --------------------------------------------------
# DATA CLASSES
# --------------------------------------------------

@dataclass
class Word:
    word: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class WordGroup:
    words: List[Word]
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def text(self) -> str:
        return " ".join(word.word for word in self.words)


@dataclass(frozen=True)
class RhythmProfile:
    words_per_second: float
    ideal_words: int
    ideal_duration: float
    max_duration: float


# --------------------------------------------------
# CREATE / NORMALIZE
# --------------------------------------------------

def create_word(raw) -> Word:
    text = clean_word(raw["word"])

    start = float(raw["start"])
    end = float(raw["end"])

    if end < start:
        end = start

    return Word(
        word=text,
        start=start,
        end=end,
    )


def create_group(words: List[Word]) -> WordGroup:
    if not words:
        raise ValueError("Nu se poate crea WordGroup fără cuvinte.")

    return WordGroup(
        words=words.copy(),
        start=words[0].start,
        end=words[-1].end,
    )


def normalized_text(value: str) -> str:
    """
    Normalizează un token doar pentru comparații lingvistice.
    Textul original rămâne neschimbat pentru subtitrare.
    """
    value = value.strip().lower()

    return re.sub(
        r'^[\s"“”„\'()[\]{}]+|[\s"“”„\'()[\]{}.,!?;:]+$',
        "",
        value,
    )


def lower(word: Word) -> str:
    return normalized_text(word.word)


def is_number(word: Word) -> bool:
    token = lower(word).replace(",", ".")
    return bool(re.fullmatch(r"\d+(?:\.\d+)?%?", token))


def is_capitalized(word: Word) -> bool:
    text = word.word.strip(" \"'“”„()[]{}.,!?;:")

    if len(text) <= 1:
        return False

    return text[0].isupper()


def is_short(word: Word) -> bool:
    return len(lower(word)) <= 2


def ends_clause(word: Word) -> bool:
    return word.word.rstrip().endswith(CLAUSE_ENDINGS)


# --------------------------------------------------
# PHRASE / ENTITY PROTECTION
# --------------------------------------------------

def boundary_breaks_common_phrase(
    words: List[Word],
    boundary: int,
) -> bool:
    """
    boundary este poziția dintre words[boundary - 1] și words[boundary].
    Returnează True dacă split-ul ar tăia o expresie cunoscută.
    """
    if boundary <= 0 or boundary >= len(words):
        return False

    normalized = [lower(word) for word in words]

    for phrase in COMMON_PHRASES:
        phrase_len = len(phrase)

        first_start = max(0, boundary - phrase_len + 1)
        last_start = min(boundary - 1, len(words) - phrase_len)

        for start in range(first_start, last_start + 1):
            end = start + phrase_len

            if not (start < boundary < end):
                continue

            if tuple(normalized[start:end]) == phrase:
                return True

    return False


def boundary_breaks_named_entity(
    words: List[Word],
    boundary: int,
) -> bool:
    if boundary <= 0 or boundary >= len(words):
        return False

    left = words[boundary - 1]
    right = words[boundary]

    # Evităm false-positive pentru cuvinte funcționale capitalizate
    # doar fiindcă sunt la început de propoziție.
    if lower(left) in COMMON_ENDERS or lower(right) in COMMON_STARTERS:
        return False

    return is_capitalized(left) and is_capitalized(right)


def boundary_breaks_number_expression(
    words: List[Word],
    boundary: int,
) -> bool:
    if boundary <= 0 or boundary >= len(words):
        return False

    left = words[boundary - 1]
    right = words[boundary]

    left_token = lower(left)
    right_token = lower(right)

    if is_number(left) and (
        right_token in NUMBER_UNITS
        or right_token == "de"
    ):
        return True

    # "20 de | ani"
    if (
        boundary >= 2
        and is_number(words[boundary - 2])
        and left_token == "de"
        and right_token in NUMBER_UNITS
    ):
        return True

    return False


def boundary_is_protected(
    words: List[Word],
    boundary: int,
) -> bool:
    return (
        boundary_breaks_common_phrase(words, boundary)
        or boundary_breaks_named_entity(words, boundary)
        or boundary_breaks_number_expression(words, boundary)
    )


# --------------------------------------------------
# SPEECH RHYTHM
# --------------------------------------------------

def local_speech_rate(
    words: List[Word],
    boundary: int,
    radius: int = LOCAL_RATE_RADIUS,
) -> float:
    """
    Calculează viteza locală în words/sec în jurul unei limite.
    Include și micile pauze dintre cuvinte, deci reflectă ritmul
    real mai bine decât media duratelor individuale.
    """
    if not words:
        return 0.0

    center = max(0, min(boundary, len(words) - 1))

    start_index = max(0, center - radius)
    end_index = min(len(words), center + radius + 1)

    sample = words[start_index:end_index]

    if not sample:
        return 0.0

    duration = sample[-1].end - sample[0].start

    if duration <= 0:
        return 0.0

    return len(sample) / duration


def rhythm_profile(
    words: List[Word],
    boundary: int,
) -> RhythmProfile:
    """
    Vorbire lentă  -> preferă 1-2 cuvinte.
    Vorbire normală -> preferă 2 cuvinte.
    Vorbire rapidă -> preferă 3 cuvinte, pentru a evita
                      schimbările excesiv de dese.
    """
    wps = local_speech_rate(words, boundary)

    if wps >= FAST_WPS:
        return RhythmProfile(
            words_per_second=wps,
            ideal_words=3,
            ideal_duration=0.95,
            max_duration=1.45,
        )

    if wps <= SLOW_WPS:
        return RhythmProfile(
            words_per_second=wps,
            ideal_words=2,
            ideal_duration=1.35,
            max_duration=MAX_DURATION,
        )

    return RhythmProfile(
        words_per_second=wps,
        ideal_words=2,
        ideal_duration=1.15,
        max_duration=1.65,
    )


# --------------------------------------------------
# NATURAL / HARD BOUNDARIES
# --------------------------------------------------

def gap_between(left: Word, right: Word) -> float:
    return max(0.0, right.start - left.end)


def is_hard_boundary(left: Word, right: Word) -> bool:
    """
    Limite pe care optimizatorul nu are voie să le traverseze:
    - final clar de propoziție
    - pauză mare
    """
    if has_sentence_end(left.word):
        return True

    return has_pause(
        left.end,
        right.start,
        HARD_PAUSE_THRESHOLD,
    )


def build_idea_spans(words: List[Word]) -> List[List[Word]]:
    """
    Împarte transcriptul în unități mari de idee înainte de
    optimizarea ritmică. Astfel caption-urile nu sar peste
    propoziții sau pauze naturale.
    """
    if not words:
        return []

    spans: List[List[Word]] = []
    current = [words[0]]

    for index in range(1, len(words)):
        previous = words[index - 1]
        word = words[index]

        if is_hard_boundary(previous, word):
            spans.append(current)
            current = [word]
        else:
            current.append(word)

    if current:
        spans.append(current)

    return spans


# --------------------------------------------------
# BOUNDARY SCORING
# --------------------------------------------------

def boundary_cost(
    words: List[Word],
    boundary: int,
) -> float:
    """
    Cost mic = loc bun de schimbat subtitrarea.
    Cost mare = ruptura ar fi nenaturală.
    """
    if boundary <= 0 or boundary >= len(words):
        return 0.0

    left = words[boundary - 1]
    right = words[boundary]

    if boundary_is_protected(words, boundary):
        return 100.0

    cost = 0.0

    pause = gap_between(left, right)

    # Pauzele sunt cele mai bune indicii pentru ritm.
    if pause >= HARD_PAUSE_THRESHOLD:
        cost -= 6.0
    elif pause >= PAUSE_THRESHOLD:
        cost -= 4.0
    elif pause >= SOFT_PAUSE_THRESHOLD:
        cost -= 1.8
    elif pause < 0.05:
        cost += 0.35

    # Punctuația semnalează finalul unei idei/clauze.
    if has_sentence_end(left.word):
        cost -= 7.0
    elif ends_clause(left):
        cost -= 2.2

    # Evităm schimbarea caption-ului înainte/după cuvinte
    # funcționale care aparțin aceleiași idei.
    if lower(left) in COMMON_ENDERS:
        cost += 3.6

    if lower(right) in COMMON_STARTERS:
        cost += 3.0

    if is_short(left) and lower(left) in COMMON_ENDERS:
        cost += 1.0

    return cost


# --------------------------------------------------
# GROUP SCORING
# --------------------------------------------------

def group_cost(
    span: List[Word],
    start: int,
    end: int,
) -> float:
    """
    Evaluează o posibilă subtitrare span[start:end].

    Costul combină:
    - viteza locală a vorbirii
    - numărul de cuvinte
    - durata caption-ului
    - calitatea limitei de după caption
    - reguli semantice simple
    """
    group_words = span[start:end]

    if not group_words:
        return float("inf")

    count = len(group_words)

    if count > MAX_WORDS:
        return float("inf")

    duration = group_words[-1].end - group_words[0].start

    profile = rhythm_profile(
        span,
        end - 1,
    )

    # O singură vorbă foarte lungă este permisă, dar un grup
    # de mai multe cuvinte nu trebuie să depășească limita.
    if count > 1 and duration > profile.max_duration:
        return float("inf")

    if duration > MAX_DURATION + 0.30:
        return float("inf")

    cost = 0.0

    # La vorbire rapidă preferăm 3 cuvinte; la vorbire
    # normală/lentă, 2 cuvinte.
    cost += abs(count - profile.ideal_words) * 1.15

    # Grupurile cu un singur cuvânt sunt permise, însă nu vrem
    # să apară continuu dacă nu există un motiv ritmic.
    if count == 1:
        cost += 1.15

        if group_words[0].duration >= 0.70:
            cost -= 0.65

        if has_sentence_end(group_words[0].word):
            cost -= 0.45

    # Ritmul vizual: caption-ul ideal rămâne suficient de mult
    # încât să fie perceput, dar nu trenează.
    cost += abs(duration - profile.ideal_duration) * 1.25

    if duration < MIN_DISPLAY_DURATION:
        cost += (MIN_DISPLAY_DURATION - duration) * 6.0

    # Evită caption-uri care încep/termină într-un loc semantic slab.
    first_token = lower(group_words[0])
    last_token = lower(group_words[-1])

    if first_token in COMMON_STARTERS and start != 0:
        cost += 1.6

    if last_token in COMMON_ENDERS and end != len(span):
        cost += 2.1

    # Costul limitei după acest grup.
    if end < len(span):
        cost += boundary_cost(span, end)

    return cost


# --------------------------------------------------
# DYNAMIC PROGRAMMING OPTIMIZER
# --------------------------------------------------

def optimize_span(span: List[Word]) -> List[WordGroup]:
    """
    Găsește combinația cu cost total minim pentru o unitate
    de idee. Fiecare caption are 1-3 cuvinte.

    Este partea de "speech rhythm": nu mai rupem mecanic după
    MAX_WORDS, ci alegem combinația care curge cel mai natural.
    """
    if not span:
        return []

    count = len(span)

    dp = [float("inf")] * (count + 1)
    choice = [None] * (count + 1)

    dp[count] = 0.0

    for start in range(count - 1, -1, -1):
        for size in range(1, MAX_WORDS + 1):
            end = start + size

            if end > count:
                break

            candidate_cost = group_cost(
                span,
                start,
                end,
            )

            if candidate_cost == float("inf"):
                continue

            total = candidate_cost + dp[end]

            if total < dp[start]:
                dp[start] = total
                choice[start] = end

    # Fallback extrem de defensiv: dacă timpii Whisper sunt
    # anormali, păstrăm câte un cuvânt în loc să pierdem text.
    if choice[0] is None:
        return [create_group([word]) for word in span]

    groups = []
    index = 0

    while index < count:
        end = choice[index]

        if end is None or end <= index:
            end = min(index + 1, count)

        groups.append(
            create_group(span[index:end])
        )

        index = end

    return groups


# --------------------------------------------------
# FINAL RULE-BASED REPAIR
# --------------------------------------------------

def can_merge(left: WordGroup, right: WordGroup) -> bool:
    if not left.words or not right.words:
        return False

    if len(left.words) + len(right.words) > MAX_WORDS:
        return False

    combined_duration = right.end - left.start

    if combined_duration > MAX_DURATION:
        return False

    pause = gap_between(left.words[-1], right.words[0])

    if pause >= HARD_PAUSE_THRESHOLD:
        return False

    return True


def merge_groups(left: WordGroup, right: WordGroup) -> WordGroup:
    return create_group(left.words + right.words)


def repair_single_word_groups(
    groups: List[WordGroup],
) -> List[WordGroup]:
    """
    Corectează cazurile rare în care DP-ul lasă un singur
    cuvânt foarte scurt între grupuri, fără să traverseze
    pauze sau limite de idee.
    """
    if len(groups) <= 1:
        return groups

    result: List[WordGroup] = []
    index = 0

    while index < len(groups):
        current = groups[index]

        if (
            len(current.words) == 1
            and current.duration < 0.65
        ):
            # Preferăm să lipim de grupul precedent dacă este sigur.
            if result and can_merge(result[-1], current):
                result[-1] = merge_groups(result[-1], current)
                index += 1
                continue

            # Altfel încercăm grupul următor.
            if (
                index + 1 < len(groups)
                and can_merge(current, groups[index + 1])
            ):
                result.append(
                    merge_groups(current, groups[index + 1])
                )
                index += 2
                continue

        result.append(current)
        index += 1

    return result


def recalculate_groups(
    groups: List[WordGroup],
) -> List[WordGroup]:
    clean_groups = []

    for group in groups:
        if not group.words:
            continue

        group.start = group.words[0].start
        group.end = group.words[-1].end

        clean_groups.append(group)

    return clean_groups


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def group_words(words):
    """
    API folosit de CaptionEngine.

    Pipeline:
        Whisper dicts
            -> Word
            -> limite mari de idee
            -> optimizare speech-rhythm prin scoring
            -> repair rule-based
            -> List[WordGroup]
    """
    if not words:
        return []

    parsed = []

    for raw_word in words:
        word = create_word(raw_word)

        if word.word:
            parsed.append(word)

    if not parsed:
        return []

    spans = build_idea_spans(parsed)

    groups: List[WordGroup] = []

    for span in spans:
        groups.extend(
            optimize_span(span)
        )

    groups = repair_single_word_groups(groups)

    return recalculate_groups(groups)
