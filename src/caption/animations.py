from enum import Enum


# --------------------------------------------------
# ANIMATION TYPES
# --------------------------------------------------

class Animation(Enum):

    NONE = "none"

    POP = "pop"

    FADE = "fade"

    ZOOM = "zoom"

    BOUNCE = "bounce"

    CAPCUT = "capcut"

    SUBMAGIC = "submagic"


# --------------------------------------------------
# GROUP ANIMATIONS
# --------------------------------------------------
#
# Aceste animații se aplică întregului Dialogue ASS.
#
# IMPORTANT:
# Nu mai facem scale pe întregul grup pentru CAPCUT / SUBMAGIC.
# Pop/zoom-ul principal este aplicat doar cuvântului activ.
#

GROUP_ANIMATION_TAGS = {

    Animation.NONE:
        "",

    Animation.FADE:
        r"{\fad(70,70)}",

    Animation.POP:
        r"{\fad(35,35)}",

    Animation.ZOOM:
        r"{\fad(35,35)}",

    Animation.BOUNCE:
        r"{\fad(30,30)}",

    Animation.CAPCUT:
        r"{\fad(30,30)}",

    Animation.SUBMAGIC:
        r"{\fad(20,20)}",
}


# --------------------------------------------------
# ACTIVE WORD ANIMATIONS
# --------------------------------------------------
#
# Aceste tag-uri se aplică numai cuvântului activ.
#
# Fiecare cuvânt activ este deja propriul event ASS,
# deci timpul 0 al animației este exact momentul în
# care începe acel cuvânt.
#

WORD_ANIMATION_TAGS = {

    Animation.NONE:
        "",

    Animation.FADE:
        r"{\fad(20,20)}",

    #
    # POP
    #
    # 90% -> 116% -> 100%
    #
    Animation.POP: (
        r"{\fscx90\fscy90"
        r"\t(0,65,\fscx116\fscy116)"
        r"\t(65,135,\fscx100\fscy100)}"
    ),

    #
    # ZOOM
    #
    # Puțin mai lent și mai fluid.
    #
    Animation.ZOOM: (
        r"{\fscx92\fscy92"
        r"\t(0,90,\fscx112\fscy112)"
        r"\t(90,180,\fscx100\fscy100)}"
    ),

    #
    # BOUNCE
    #
    # Overshoot + mic recul.
    #
    Animation.BOUNCE: (
        r"{\fscx88\fscy88"
        r"\t(0,60,\fscx118\fscy118)"
        r"\t(60,115,\fscx97\fscy97)"
        r"\t(115,180,\fscx100\fscy100)}"
    ),

    #
    # CAPCUT
    #
    # Pop rapid și curat.
    #
    Animation.CAPCUT: (
        r"{\fscx90\fscy90"
        r"\t(0,55,\fscx114\fscy114)"
        r"\t(55,120,\fscx100\fscy100)}"
    ),

    #
    # SUBMAGIC
    #
    # Puțin mai agresiv decât CAPCUT.
    #
    Animation.SUBMAGIC: (
        r"{\fscx88\fscy88"
        r"\t(0,50,\fscx117\fscy117)"
        r"\t(50,105,\fscx102\fscy102)"
        r"\t(105,160,\fscx100\fscy100)}"
    ),
}


# --------------------------------------------------
# RESET
# --------------------------------------------------

WORD_RESET_TAG = (
    r"{\fscx100"
    r"\fscy100}"
)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def normalize_animation(animation) -> Animation:
    """
    Acceptă atât Animation cât și string-uri:

    "capcut"
    "submagic"
    "pop"
    "zoom"
    """

    if isinstance(
        animation,
        Animation
    ):
        return animation

    if isinstance(
        animation,
        str
    ):

        value = animation.strip().lower()

        for item in Animation:

            if item.value == value:
                return item

    return Animation.NONE


# --------------------------------------------------
# GROUP ANIMATION
# --------------------------------------------------

def build_animation(
    animation
) -> str:
    """
    Animație pentru întregul grup.

    Numele funcției este păstrat pentru
    compatibilitate cu ASSWriter.
    """

    animation = normalize_animation(
        animation
    )

    return GROUP_ANIMATION_TAGS.get(
        animation,
        ""
    )


# --------------------------------------------------
# WORD ANIMATION
# --------------------------------------------------

def build_word_animation(
    animation
) -> str:
    """
    Returnează animația pentru cuvântul activ.

    Exemplu:

        CAPCUT:
        90% -> 114% -> 100%

    Această funcție trebuie apelată din karaoke.py,
    în active_word().
    """

    animation = normalize_animation(
        animation
    )

    return WORD_ANIMATION_TAGS.get(
        animation,
        ""
    )


def build_word_reset() -> str:
    """
    Resetează scale-ul după cuvântul activ,
    astfel încât animația să nu afecteze
    cuvintele următoare.
    """

    return WORD_RESET_TAG
