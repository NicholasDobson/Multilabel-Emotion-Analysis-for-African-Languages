import argparse
import logging
import os
import time
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import google.generativeai as genai
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, util
import torch

from utils import EMOTIONS, TEXT_COL, load_language_data

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIMILARITY_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")
    genai.configure(api_key=api_key)

    return genai.GenerativeModel('gemini-3.1-flash-lite')

def batch_paraphrase(texts: List[str], lang: str, model, batch_size: int = 20) -> List[str]:
    # use Gemini in batches because one at a time is too slow
    all_paraphrases = []

    from tqdm import tqdm
    for i in tqdm(range(0, len(texts), batch_size), desc="Gemini Paraphrasing"):
        chunk = texts[i : i + batch_size]

        numbered_texts = "\n".join([f"{idx+1}. {text}" for idx, text in enumerate(chunk)])

        prompt = f"""
        You are an expert in the '{lang}' language.
        I will provide a numbered list of {len(chunk)} sentences.
        Paraphrase each sentence in '{lang}' while strictly maintaining its exact original meaning and emotional tone.

        Sentences to paraphrase:
        {numbered_texts}
        """

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=list[str]
                )
            )

            batch_results = json.loads(response.text)

            if len(batch_results) != len(chunk):
                logger.warning(f"Batch size mismatch! Expected {len(chunk)}, got {len(batch_results)}. Skipping batch.")
                all_paraphrases.extend(chunk)
            else:
                all_paraphrases.extend(batch_results)

        except Exception as e:
            logger.error(f"API Error on batch {i}: {e}. Skipping batch.")
            all_paraphrases.extend(chunk)

        time.sleep(4.1)

    return all_paraphrases

def main():
    parser = argparse.ArgumentParser(description="Stage 2b: Cloud Paraphrasing via Gemini")
    parser.add_argument("--lang", required=True, help="Language code, e.g. amh, afr")
    parser.add_argument("--batch_size", type=int, default=20, help="Sentences per API call")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold")
    args = parser.parse_args()

    gemini_model = setup_gemini()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_language_data(args.lang)
    train_texts, train_labels = zip(*splits["train"])
    train_texts, train_labels = list(train_texts), list(train_labels)

    logger.info(f"Starting batched API paraphrasing for {len(train_texts)} sentences...")
    para_texts = batch_paraphrase(train_texts, args.lang, gemini_model, args.batch_size)

    logger.info(f"Loading local similarity model...")
    sim_model = SentenceTransformer(SIMILARITY_MODEL, device=device)

    orig_emb = sim_model.encode(train_texts, convert_to_tensor=True, show_progress_bar=False)
    aug_emb = sim_model.encode(para_texts, convert_to_tensor=True, show_progress_bar=False)
    similarities = util.cos_sim(orig_emb, aug_emb).diagonal().cpu().numpy()

    kept_samples = []
    for i, sim in enumerate(similarities):
        if float(sim) >= args.threshold and para_texts[i].strip() != train_texts[i].strip():
            kept_samples.append((para_texts[i], train_labels[i]))

    logger.info(f"Paraphrasing kept {len(kept_samples)}/{len(train_texts)} samples (similarity >= {args.threshold})")

    aug_dict = DatasetDict({
        "train": Dataset.from_dict({
            TEXT_COL: list(train_texts) + [s[0] for s in kept_samples],
            **{e: list(zip(*train_labels))[j] + tuple(s[1][j] for s in kept_samples) for j, e in enumerate(EMOTIONS)}
        }),
        "validation": Dataset.from_dict({TEXT_COL: [x[0] for x in splits["validation"]], **{e: [x[1][j] for x in splits["validation"]] for j, e in enumerate(EMOTIONS)}}),
        "test": Dataset.from_dict({TEXT_COL: [x[0] for x in splits["test"]], **{e: [x[1][j] for x in splits["test"]] for j, e in enumerate(EMOTIONS)}}),
    })

    out_path = Path("data") / f"{args.lang}_gemini_augmented"
    aug_dict.save_to_disk(str(out_path))
    logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
