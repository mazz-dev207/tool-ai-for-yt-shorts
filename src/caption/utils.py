import re



def ass_time(seconds: float) -> str:
    """
    Convertește secunde în format ASS:

    H:MM:SS.CC
    """

    if seconds < 0:
        seconds = 0


    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    centiseconds = int(
        round(
            (seconds - int(seconds)) * 100
        )
    )


    if centiseconds >= 100:

        centiseconds = 0

        secs += 1


    if secs >= 60:

        secs = 0

        minutes += 1


    if minutes >= 60:

        minutes = 0

        hours += 1


    return (
        f"{hours}:"
        f"{minutes:02}:"
        f"{secs:02}."
        f"{centiseconds:02}"
    )



def clean_word(word: str) -> str:
    """
    Curăță cuvintele venite din Whisper.
    """

    if not word:

        return ""


    return (
        word
        .strip()
        .replace("\n", " ")
    )



def remove_punctuation(word: str) -> str:
    """
    Elimină semnele de punctuație
    pentru efecte karaoke.
    """

    return re.sub(

        r"[.,!?;:\"'(){}\[\]]",

        "",

        word

    )



def has_sentence_end(word: str) -> bool:
    """
    Verifică final de propoziție.
    """

    if not word:

        return False


    return bool(

        re.search(

            r"[.!?][\"']?$",

            word

        )

    )



def has_pause(
    previous_end: float,
    current_start: float,
    threshold: float = 0.35
) -> bool:
    """
    Verifică pauza dintre cuvinte.
    """

    return (

        current_start - previous_end

    ) >= threshold



def is_punctuation(word: str) -> bool:
    """
    Verifică dacă este doar punctuație.
    """

    if not word:

        return False


    return bool(

        re.fullmatch(

            r"[.,!?;:]+",

            word

        )

    )



def clamp(
    value,
    minimum,
    maximum
):
    """
    Limitează o valoare.
    """

    return max(

        minimum,

        min(

            value,

            maximum

        )

    )