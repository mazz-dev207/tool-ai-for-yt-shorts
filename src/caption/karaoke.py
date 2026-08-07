from src.caption.grouping import WordGroup, Word
from src.caption.animations import (
    build_word_animation,
    build_word_reset,
)
from src.config import CAPTION_ANIMATION


# --------------------------------------------------
# COLORS
# --------------------------------------------------

DEFAULT_COLOR = "&H00FFFFFF"
ACTIVE_COLOR = "&H0000FFFF"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def escape_ass_text(text: str) -> str:
    """
    Escape pentru caractere speciale ASS.
    """
    if not text:
        return ""

    text = text.replace("\\", r"\\")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")

    return text


def safe_duration(
    start: float,
    end: float
) -> float:
    """
    Evită durate negative sau zero
    provenite din timestamp-uri Whisper imperfecte.
    """
    return max(
        0.01,
        end - start
    )


# --------------------------------------------------
# WORD TEXT
# --------------------------------------------------

def normal_word(
    word: Word
) -> str:
    """
    Cuvânt normal:
    - alb
    - scale 100%
    - fără animație
    """
    text = escape_ass_text(
        word.word
    )

    return (
        f"{{"
        f"\\c{DEFAULT_COLOR}"
        f"\\fscx100"
        f"\\fscy100"
        f"}}"
        f"{text}"
    )


def active_word(
    word: Word,
    animation=CAPTION_ANIMATION
) -> str:
    """
    Cuvântul activ:
    - galben
    - pop / zoom / bounce
    - reset la 100%
    """
    text = escape_ass_text(
        word.word
    )

    word_animation = build_word_animation(
        animation
    )

    reset = build_word_reset()

    return (
        f"{{\\c{ACTIVE_COLOR}}}"
        f"{word_animation}"
        f"{text}"
        f"{reset}"
    )


# --------------------------------------------------
# BUILD FRAME
# --------------------------------------------------

def build_active_frame(
    group: WordGroup,
    active_index: int,
    animation=CAPTION_ANIMATION
) -> str:
    """
    Construiește întregul grup.
    Doar cuvântul activ este galben și animat.
    """
    parts = []

    for index, word in enumerate(
        group.words
    ):
        if index == active_index:
            parts.append(
                active_word(
                    word,
                    animation
                )
            )
        else:
            parts.append(
                normal_word(word)
            )

    return " ".join(parts)


# --------------------------------------------------
# WORD EVENTS
# --------------------------------------------------

def build_word_events(
    group: WordGroup,
    animation=CAPTION_ANIMATION
):
    """
    Generează câte un event ASS pentru fiecare cuvânt.

    Event-ul curent continuă până la începutul
    următorului cuvânt pentru a evita flicker-ul.
    """
    events = []

    if not group.words:
        return events

    for index, word in enumerate(
        group.words
    ):
        start = word.start

        if index < len(group.words) - 1:
            next_word = group.words[
                index + 1
            ]

            end = max(
                word.end,
                next_word.start
            )
        else:
            end = max(
                word.end,
                group.end
            )

        duration = safe_duration(
            start,
            end
        )

        end = start + duration

        text = build_active_frame(
            group,
            index,
            animation
        )

        events.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "active_index": index,
                "word": word.word
            }
        )

    return events


# --------------------------------------------------
# FALLBACK
# --------------------------------------------------

def build_plain_group(
    group: WordGroup
) -> str:
    """
    Construiește grupul fără highlight sau animație.
    """
    return " ".join(
        normal_word(word)
        for word in group.words
    )


# --------------------------------------------------
# COMPATIBILITY
# --------------------------------------------------

def build_karaoke(
    group: WordGroup
) -> str:
    """
    Compatibilitate cu codul vechi.
    """
    return build_plain_group(
        group
    )
