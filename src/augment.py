import argparse
import logging
import sys
import gc
from pathlib import Path

import numpy as np
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

NLLB_MODEL = "facebook/nllb-200-distilled-600M" 
AYA_MODEL = "CohereForAI/aya-101"

NLLB_LANG_MAP = {
    "afr": "afr_Latn",
    "amh": "amh_Ethi",
    "swa": "swa_Latn",
    "xho": "xho_Latn",
    "zul": "zul_Latn",
    "yor": "yor_Latn",
    "en": "eng_Latn"
}

def _free_memory(model, tokenizer):
    # clean up the mess we made on the GPU
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def backtranslate(texts: list, lang: str, device, batch_size: int):
    # taking a scenic detour through English to get new perspectives
    nllb_lang = NLLB_LANG_MAP.get(lang)
    if not nllb_lang:
        logger.error(f"Language {lang} not in NLLB_LANG_MAP.")
        return None, False

    logger.info(f"Loading NLLB model: {NLLB_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, revision="refs/pr/45")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, revision="refs/pr/45", torch_dtype=torch.float16).to(device)
    model.eval()

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
    logger.info(f"Loading Aya model: {AYA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(AYA_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(AYA_MODEL, torch_dtype=torch.float16).to(device)
    model.eval()

    results = []
    logger.info(f"Generating paraphrases using Aya...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Aya Paraphrasing"):
        chunk = texts[i : i + batch_size]

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

def filter_by_similarity(
    originals: list, augmented: list, sim_model: SentenceTransformer, threshold: float
) -> list:
    orig_emb = sim_model.encode(originals, convert_to_tensor=True, show_progress_bar=False)
    aug_emb = sim_model.encode(augmented, convert_to_tensor=True, show_progress_bar=False)
    similarities = util.cos_sim(orig_emb, aug_emb).diagonal().cpu().numpy()
    return [i for i, s in enumerate(similarities) if float(s) >= threshold]

def identify_minority_samples(train_texts, train_labels):
    # finding the underdogs in our dataset that need some extra love
    labels_array = np.array(train_labels)
    label_counts = labels_array.sum(axis=0)
    median_count = np.median(label_counts)

    minority_mask = label_counts < median_count
    minority_label_names = [EMOTIONS[i] for i, m in enumerate(minority_mask) if m]
    logger.info(f"Label counts: {dict(zip(EMOTIONS, label_counts.astype(int).tolist()))}")
    logger.info(f"Median count: {median_count:.0f}")
    logger.info(f"Minority labels (below median): {minority_label_names}")

    minority_indices = [
        i for i, lbls in enumerate(train_labels)
        if any(lbls[j] == 1 for j in range(len(EMOTIONS)) if minority_mask[j])
    ]

    minority_texts = [train_texts[i] for i in minority_indices]
    minority_labels = [train_labels[i] for i in minority_indices]

    logger.info(f"Augmenting {len(minority_texts)}/{len(train_texts)} minority samples only")

    return minority_texts, minority_labels, minority_mask

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
            if aug_texts[i].strip() != texts[i].strip():
                kept.append((aug_texts[i], labels[i]))

    return kept

def pairs_to_hf_dataset(pairs: list) -> Dataset:
    if not pairs:
        return Dataset.from_dict({TEXT_COL: [], **{e: [] for e in EMOTIONS}})
    texts, labels = zip(*pairs)
    rows = {TEXT_COL: list(texts)}
    for j, emotion in enumerate(EMOTIONS):
        rows[emotion] = [int(lbl[j]) for lbl in labels]
    return Dataset.from_dict(rows)

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
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold")
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument(
        "--verify_labels", action="store_true",
        help="Use Gemini LLM to verify label preservation on augmented samples (slower but cleaner)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Lang: {args.lang} | Methods: {args.methods} | Device: {device}")

    splits = load_language_data(args.lang)
    if not splits["train"]:
        raise ValueError(f"No training data for lang={args.lang}")

    train_texts, train_labels = zip(*splits["train"])
    train_texts, train_labels = list(train_texts), list(train_labels)

    minority_texts, minority_labels, minority_mask = identify_minority_samples(
        train_texts, train_labels
    )

    logger.info(f"Loading similarity model: {SIMILARITY_MODEL}")
    sim_model = SentenceTransformer(SIMILARITY_MODEL, device=device)

    new_samples = augment_training_split(
        minority_texts, minority_labels, args.lang, args.methods,
        sim_model, device, args.threshold, args.batch_size,
    )

    if args.verify_labels and new_samples:
        logger.info(f"Running Gemini label verification on {len(new_samples)} augmented samples...")
        from filter_gemini import verify_labels_batch
        from paraphrase_gemini import setup_gemini

        gemini_model = setup_gemini()
        aug_texts_to_verify = [s[0] for s in new_samples]
        aug_labels_to_verify = [s[1] for s in new_samples]

        verified = verify_labels_batch(
            aug_texts_to_verify, aug_labels_to_verify,
            args.lang, gemini_model, EMOTIONS,
        )

        pre_filter_count = len(new_samples)
        new_samples = [s for s, v in zip(new_samples, verified) if v]
        logger.info(
            f"Label verification: {len(new_samples)}/{pre_filter_count} samples passed "
            f"({pre_filter_count - len(new_samples)} rejected)"
        )

    original_n = len(train_texts)
    augmented_n = original_n + len(new_samples)

    if augmented_n <= original_n:
         logger.warning("No augmented samples passed the filters.")
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
