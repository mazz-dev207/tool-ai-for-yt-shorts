from pathlib import Path
import json
import sys
import time

import ollama

from src.config import (
    HIGHLIGHTS_DIR,
    TEMP_DIR,
    OLLAMA_MODEL
)

from src.logger import (
    info,
    success
)


MIN_SCORE = 70


def analyze_window(window):

    text = ""

    for segment in window["segments"]:

        text += (
            f"[{segment['start']:.1f}s - "
            f"{segment['end']:.1f}s] "
            f"{segment['text']}\n"
        )


    prompt = f"""
You are an expert YouTube Shorts editor.

Analyze this transcript and decide how viral this moment could be.

Evaluate:

- strong hook
- curiosity
- emotional impact
- storytelling
- educational value
- audience retention potential

Return ONLY valid JSON.

Format:

{{
    "title": "short title",
    "score": 85
}}

Score:
0 = boring
100 = extremely engaging

Transcript:

{text}
"""


    start_time = time.time()


    response = ollama.chat(

        model=OLLAMA_MODEL,

        stream=False,

        messages=[

            {
                "role": "system",
                "content":
                "Return only valid JSON. No explanations."
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        format="json",

        options={

            "temperature": 0.2,

            "think": False

        }

    )


    elapsed = (
        time.time()
        -
        start_time
    )


    info(
        f"Ollama răspuns în {elapsed:.2f}s"
    )


    content = (
        response["message"]["content"]
        .strip()
    )


    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError:

        info(
            "Răspuns JSON invalid. Se ignoră."
        )

        result = {

            "title":
            "Untitled",

            "score":
            0

        }


    if "title" not in result:

        result["title"] = (
            "Untitled"
        )


    if "score" not in result:

        result["score"] = 0


    result["start"] = (
        window["start"]
    )

    result["end"] = (
        window["end"]
    )


    return result
def overlap(a, b):

    return (

        a["start"] < b["end"]

        and

        a["end"] > b["start"]

    )



def select_highlights(video_name):


    chunk_file = (

        TEMP_DIR

        /

        f"{video_name}_chunks.json"

    )


    if not chunk_file.exists():

        raise FileNotFoundError(

            chunk_file

        )


    with open(

        chunk_file,

        encoding="utf-8"

    ) as f:

        windows = json.load(f)



    info(

        f"Analizez {len(windows)} ferestre..."

    )


    results = []


    for index, window in enumerate(

        windows,

        start=1

    ):

        info(

            f"Fereastră {index}/{len(windows)}"

        )


        try:


            result = analyze_window(

                window

            )


            info(

                f"Score: {result['score']} | "

                f"{result['title']}"

            )


            if result["score"] >= MIN_SCORE:


                results.append(

                    result

                )


        except Exception as e:


            info(

                f"Eroare la fereastra {index}: {e}"

            )



    if not results:


        info(

            "Niciun highlight nu a trecut scorul minim."

        )



    results.sort(

        key=lambda x: x["score"],

        reverse=True

    )



    final = []



    for clip in results:


        duplicated = False


        for selected in final:


            if overlap(

                clip,

                selected

            ):


                duplicated = True

                break



        if not duplicated:


            final.append(

                clip

            )



    HIGHLIGHTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    output = (

        HIGHLIGHTS_DIR

        /

        f"{video_name}.json"

    )



    with open(

        output,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            final,

            f,

            indent=4,

            ensure_ascii=False

        )



    success(

        f"Au rămas {len(final)} clipuri."

    )


    success(

        f"Highlights salvate: {output}"

    )


    return output





if __name__ == "__main__":


    if len(sys.argv) < 2:


        print(

            'Utilizare: python src/highlight_selector.py "video_name"'

        )

        sys.exit(1)



    try:


        video_name = " ".join(

            sys.argv[1:]

        )


        select_highlights(

            video_name

        )


    except Exception as e:


        print()

        print(

            f"Eroare: {e}"

        )

        sys.exit(1)