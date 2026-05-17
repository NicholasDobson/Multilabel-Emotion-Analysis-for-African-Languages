"""
Stage 2: Modular data augmentation pipeline for multilabel emotion classification.

Usage:
  python src/augment.py --lang amh --methods bt
  python src/augment.py --lang afr --methods bt para
"""

import argparse
import logging
import sys
import gc
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import EMOTIONS, TEXT_COL, load_language_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIMILARITY_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Model Configurations ──────────────────────────────────────────────────────

# Meta's NLLB for Back-Translation
NLLB_MODEL = "facebook/nllb-200-distilled-600M" # Upgrade to "facebook/nllb-200-1.3B" if VRAM allows
# Cohere's Aya for Paraphrasing
AYA_MODEL = "CohereForAI/aya-101"

# NLLB uses FLORES-200 language codes. Map your simple codes to these.
NLLB_LANG_MAP = {
    "afr": "afr_Latn",
    "amh": "amh_Ethi",
    "swa": "swa_Latn",
    "xho": "xho_Latn",
    "zul": "zul_Latn",
    "yor": "yor_Latn",
    "en": "eng_Latn"
}

# ── Translation & Generation Helpers ──────────────────────────────────────────

def _free_memory(model, tokenizer):
    """Deletes model from memory and clears GPU cache."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Augmentation methods ──────────────────────────────────────────────────────

def backtranslate(texts: list, lang: str, device, batch_size: int):
    """
    Translate texts lang → English → lang using NLLB.
    """
    nllb_lang = NLLB_LANG_MAP.get(lang)
    if not nllb_lang:
        logger.error(f"Language {lang} not in NLLB_LANG_MAP.")
        return None, False

    logger.info(f"Loading NLLB model: {NLLB_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, revision="refs/pr/45")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, revision="refs/pr/45", torch_dtype=torch.float16).to(device)
    model.eval()

    # 1. Translate Source -> English
    tokenizer.src_lang = nllb_lang
    eng_target_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    en_texts = []

    logger.info(f"Translating {lang} -> eng_Latn...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Source -> English"):
        chunk = texts[i : i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=eng_target_id, max_length=512)
        en_texts.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    # 2. Translate English -> Source
    tokenizer.src_lang = "eng_Latn"
    src_target_id = tokenizer.convert_tokens_to_ids(nllb_lang)
    bt_texts = []

    logger.info(f"Translating eng_Latn -> {lang}...")
    for i in tqdm(range(0, len(en_texts), batch_size), desc="English -> Source"):
        chunk = en_texts[i : i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=src_target_id, max_length=512)
        bt_texts.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    _free_memory(model, tokenizer)
    return bt_texts, True


def paraphrase(texts: list, lang: str, device, batch_size: int):
    """
    Generate paraphrases using Cohere's Aya-101.
    """
    logger.info(f"Loading Aya model: {AYA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(AYA_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(AYA_MODEL, torch_dtype=torch.float16).to(device)
    model.eval()

    results = []
    logger.info(f"Generating paraphrases using Aya...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Aya Paraphrasing"):
        chunk = texts[i : i + batch_size]

        # Format prompts specifically for Aya
        prompts = [
            f"Paraphrase the following text in its original language. Maintain the exact meaning and emotional tone:\n{text}"
            for text in chunk
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=True)
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    _free_memory(model, tokenizer)
    return results


# ── Quality filter ────────────────────────────────────────────────────────────

def filter_by_similarity(
    originals: list, augmented: list, sim_model: SentenceTransformer, threshold: float
) -> list:
    """Return indices where cosine similarity(original, augmented) >= threshold."""
    orig_emb = sim_model.encode(originals, convert_to_tensor=True, show_progress_bar=False)
    aug_emb = sim_model.encode(augmented, convert_to_tensor=True, show_progress_bar=False)
    similarities = util.cos_sim(orig_emb, aug_emb).diagonal().cpu().numpy()
    return [i for i, s in enumerate(similarities) if float(s) >= threshold]


# ── Core augmentation loop ────────────────────────────────────────────────────

def augment_training_split(
    texts: list,
    labels: list,
    lang: str,
    methods: list,
    sim_model: SentenceTransformer,
    device,
    threshold: float,
    batch_size: int,
) -> list:
    """
    Apply each enabled augmentation method, filter by similarity, and return
    a list of (text, labels) tuples for the kept augmented samples.
    """
    kept = []
    for method in methods:
        logger.info(f"Running method: {method}")
        if method == "bt":
            aug_texts, ok = backtranslate(texts, lang, device, batch_size)
            if not ok:
                logger.warning(f"Back-translation skipped for lang={lang} (model unavailable)")
                continue
        elif method == "para":
            aug_texts = paraphrase(texts, lang, device, batch_size)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

        passed = filter_by_similarity(texts, aug_texts, sim_model, threshold)
        logger.info(f"  {method}: {len(passed)}/{len(texts)} samples kept (similarity >= {threshold})")
        for i in passed:
            # Prevent adding duplicates if the output is exactly the same as the input
            if aug_texts[i].strip() != texts[i].strip():
                kept.append((aug_texts[i], labels[i]))

    return kept


# ── Dataset serialisation ─────────────────────────────────────────────────────

def pairs_to_hf_dataset(pairs: list) -> Dataset:
    """Convert a list of (text, labels) tuples to a HuggingFace Dataset."""
    if not pairs:
        return Dataset.from_dict({TEXT_COL: [], **{e: [] for e in EMOTIONS}})
    texts, labels = zip(*pairs)
    rows = {TEXT_COL: list(texts)}
    for j, emotion in enumerate(EMOTIONS):
        rows[emotion] = [int(lbl[j]) for lbl in labels]
    return Dataset.from_dict(rows)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2: data augmentation pipeline")
    parser.add_argument("--lang", required=True, help="Language config code, e.g. amh, afr")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bt", "para"],
        default=["bt"],
        help="Augmentation methods: bt (back-translation), para (paraphrase)",
    )
    # Reduced default threshold to 0.75 as cross-lingual similarity models are often strict
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold")
    parser.add_argument("--batch_size", type=int, default=16) # Reduced default batch size to save VRAM
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Lang: {args.lang} | Methods: {args.methods} | Device: {device}")

    splits = load_language_data(args.lang)
    if not splits["train"]:
        raise ValueError(f"No training data for lang={args.lang}")

    train_texts, train_labels = zip(*splits["train"])
    train_texts, train_labels = list(train_texts), list(train_labels)

    logger.info(f"Loading similarity model: {SIMILARITY_MODEL}")
    sim_model = SentenceTransformer(SIMILARITY_MODEL, device=device)

    new_samples = augment_training_split(
        train_texts, train_labels, args.lang, args.methods,
        sim_model, device, args.threshold, args.batch_size,
    )

    original_n = len(train_texts)
    augmented_n = original_n + len(new_samples)

    if augmented_n <= original_n:
         logger.warning("No augmented samples passed the similarity filter.")
    else:
         logger.info(f"Train split: {original_n} → {augmented_n} samples ({augmented_n / original_n:.2f}x)")

    aug_dict = DatasetDict({
        "train": pairs_to_hf_dataset(list(zip(train_texts, train_labels)) + new_samples),
        "validation": pairs_to_hf_dataset(splits["validation"]),
        "test": pairs_to_hf_dataset(splits["test"]),
    })

    out_path = Path("data") / f"{args.lang}_augmented"
    aug_dict.save_to_disk(str(out_path))
    logger.info(f"Augmented dataset saved to {out_path}")


if __name__ == "__main__":
    main()

