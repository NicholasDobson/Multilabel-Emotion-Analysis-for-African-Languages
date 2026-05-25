import json
import logging
import time
from typing import List

from dotenv import load_dotenv
import google.generativeai as genai

from paraphrase_gemini import setup_gemini
from utils import EMOTIONS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def verify_labels_batch(
    texts: List[str],
    original_labels: List[List[int]],
    lang: str,
    model,
    emotions: List[str] = EMOTIONS,
    batch_size: int = 10,
) -> List[bool]:
    results = []

    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i : i + batch_size]
        chunk_labels = original_labels[i : i + batch_size]

        label_descriptions = [
            ", ".join(emotions[j] for j in range(len(emotions)) if lbl[j] == 1)
            or "neutral"
            for lbl in chunk_labels
        ]

        numbered = "\n".join(
            f'{k+1}. Text: "{t}" | Expected emotions: {d}'
            for k, (t, d) in enumerate(zip(chunk_texts, label_descriptions))
        )

        prompt = f"""You are an expert in the '{lang}' language and emotion analysis.
For each numbered text below, answer ONLY with true or false:
- true  = the text clearly expresses the listed emotions
- false = the text does NOT express those emotions, or expresses different ones

{numbered}

Respond with a JSON array of {len(chunk_texts)} booleans, e.g. [true, false, true]"""

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=list[bool],
                ),
            )
            batch_results = json.loads(response.text)

            if len(batch_results) == len(chunk_texts):
                results.extend(batch_results)
            else:
                logger.warning(
                    f"Gemini returned {len(batch_results)} results for "
                    f"{len(chunk_texts)} texts — rejecting batch"
                )
                results.extend([False] * len(chunk_texts))
        except Exception as e:
            logger.error(f"Gemini verify error: {e}")
            results.extend([False] * len(chunk_texts))

        time.sleep(4.1)

    return results
