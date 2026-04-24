"""
Stage 2: Modular data augmentation pipeline for multilabel emotion classification.

Usage:
  python src/augment.py --lang amh --methods bt
  python src/augment.py --lang afr --methods bt para
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, str(Path(__file__).parent))
from utils import EMOTIONS, TEXT_COL, load_language_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIMILARITY_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# TO-DO: find and verify a multilingual paraphrase model.
# google/mt5-small is a placeholder — it is NOT fine-tuned for paraphrasing
# and will produce low-quality output. Replace with a verified multilingual
# paraphrase model (e.g. a community fine-tune of mT5 on a paraphrase corpus)
# before using --methods para in experiments.
PARAPHRASE_MODEL = "google/mt5-small"  # TO-DO: replace with verified model


# ── Translation helpers ───────────────────────────────────────────────────────

def _load_marian(src: str, tgt: str, device):
    """Load a Helsinki-NLP MarianMT model. Returns (model, tokenizer) or None."""
    from transformers import MarianMTModel, MarianTokenizer
    model_id = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        tok = MarianTokenizer.from_pretrained(model_id)
        mdl = MarianMTModel.from_pretrained(model_id).to(device)
        mdl.eval()
        logger.info(f"Loaded {model_id}")
        return mdl, tok
    except Exception as e:
        logger.warning(f"MarianMT model {model_id} not available: {e}")
        return None


def _translate_batch(texts: list, model, tokenizer, device, batch_size: int) -> list:
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs)
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return results


# ── Augmentation methods ──────────────────────────────────────────────────────

def backtranslate(texts: list, lang: str, device, batch_size: int):
    """
    Translate texts lang → English → lang.
    Returns (augmented_texts, True) or (None, False) if a model is unavailable.
    """
    fwd = _load_marian(lang, "en", device)
    if fwd is None:
        return None, False
    rev = _load_marian("en", lang, device)
    if rev is None:
        return None, False

    en_texts = _translate_batch(texts, *fwd, device, batch_size)
    bt_texts = _translate_batch(en_texts, *rev, device, batch_size)
    return bt_texts, True


def paraphrase(texts: list, device, batch_size: int):
    """
    Generate paraphrases with a seq2seq model.
    TO-DO: replace PARAPHRASE_MODEL with a verified multilingual paraphrase
    model before using this method. Current placeholder will not produce
    useful paraphrases.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as HFTokenizer
    tok = HFTokenizer.from_pretrained(PARAPHRASE_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to(device)
    mdl.eval()

    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tok(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=True)
        results.extend(tok.batch_decode(out, skip_special_tokens=True))
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
            aug_texts = paraphrase(texts, device, batch_size)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

        passed = filter_by_similarity(texts, aug_texts, sim_model, threshold)
        logger.info(f"  {method}: {len(passed)}/{len(texts)} samples kept (similarity >= {threshold})")
        for i in passed:
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
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Lang: {args.lang} | Methods: {args.methods} | Device: {device}")

    # Load original data
    splits = load_language_data(args.lang)
    if not splits["train"]:
        raise ValueError(f"No training data for lang={args.lang}")

    train_texts, train_labels = zip(*splits["train"])
    train_texts, train_labels = list(train_texts), list(train_labels)

    # Load similarity model once; reused across all methods
    logger.info(f"Loading similarity model: {SIMILARITY_MODEL}")
    sim_model = SentenceTransformer(SIMILARITY_MODEL)

    # Augment
    new_samples = augment_training_split(
        train_texts, train_labels, args.lang, args.methods,
        sim_model, device, args.threshold, args.batch_size,
    )

    original_n = len(train_texts)
    augmented_n = original_n + len(new_samples)
    assert augmented_n > original_n, (
        f"No augmented samples passed the similarity filter — "
        f"augmented dataset ({augmented_n}) is not larger than original ({original_n})"
    )
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
