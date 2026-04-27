"""
Shared utilities for multilabel emotion classification.
Treat metric functions as frozen after Stage 1 — do not modify compute_metrics.
"""

import logging
import unicodedata
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score, jaccard_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

EMOTIONS = ["anger", "fear", "surprise", "sadness", "joy", "disgust"]

DATASET_IDS = {
    "brighter": "brighter-dataset/BRIGHTER-emotion-categories",
    "ethioemo": "Tadesse/EthioEmo",
}

TEXT_COL = "text"
# Both datasets load via language config (e.g. "amh", "afr"); LANG_COL is only
# used as a fallback filter if config-based loading fails.
LANG_COL = "language"


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _extract_labels(example: dict) -> list:
    return [int(bool(example.get(emotion, 0))) for emotion in EMOTIONS]


def _resolve_splits(raw_dataset):
    """Return (train, val, test) HuggingFace Dataset objects from a DatasetDict."""
    available = set(raw_dataset.keys())

    # BRIGHTER uses "dev" instead of "validation"
    val_key = "validation" if "validation" in available else "dev" if "dev" in available else None

    if "train" in available and val_key and "test" in available:
        return raw_dataset["train"], raw_dataset[val_key], raw_dataset["test"]

    if "train" in available:
        full = raw_dataset["train"]
        n = len(full)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        idx = list(range(n))
        return (
            full.select(idx[:train_end]),
            full.select(idx[train_end:val_end]),
            full.select(idx[val_end:]),
        )

    raise ValueError(f"Dataset has no 'train' split; found: {available}")


def _load_one_dataset(ds_id: str, lang: str):
    """
    Try to load a single HuggingFace dataset for the given language.
    Returns list of (train_rows, val_rows, test_rows) or None on failure.
    Each row list contains (text, labels) tuples.
    """
    from datasets import load_dataset

    raw = None

    # Strategy 1: dataset has per-language configs
    try:
        raw = load_dataset(ds_id, lang)
        logger.info(f"Loaded {ds_id} with config '{lang}'")
    except Exception as e:
        err = str(e)
        # If the error says this language config doesn't exist, skip the dataset
        # entirely. Falling through to Strategy 2 would trigger an interactive
        # prompt asking which config to load.
        if "available configs" in err or "Config name is missing" in err:
            logger.warning(f"'{lang}' not available in {ds_id} — skipping")
            return None

    # Strategy 2: load full dataset and filter by language column.
    # Only reached if Strategy 1 failed for a non-config reason (e.g. a dataset
    # that has no configs and uses a language column instead).
    if raw is None:
        try:
            full = load_dataset(ds_id)
            first_split = full[next(iter(full))]
            if LANG_COL in first_split.column_names:
                raw = {k: v.filter(lambda x: x[LANG_COL] == lang) for k, v in full.items()}
                logger.info(f"Loaded {ds_id} and filtered by {LANG_COL}='{lang}'")
            else:
                raw = full
                logger.warning(
                    f"{ds_id}: no '{LANG_COL}' column found; loading without language filter"
                )
        except Exception as e:
            logger.warning(f"Could not load {ds_id} for lang={lang}: {e}")
            return None

    try:
        train_ds, val_ds, test_ds = _resolve_splits(raw)
    except Exception as e:
        logger.warning(f"Could not resolve splits for {ds_id}: {e}")
        return None

    result = []
    for split_ds in (train_ds, val_ds, test_ds):
        rows = []
        for example in split_ds:
            if TEXT_COL not in example:
                continue
            text = normalize_text(str(example[TEXT_COL]))
            labels = _extract_labels(example)
            rows.append((text, labels))
        result.append(rows)

    return result  # [train_rows, val_rows, test_rows]


def load_language_data(lang: str) -> dict:
    """
    Load and merge data for a language from all configured datasets.
    Returns {"train": [...], "validation": [...], "test": [...]}
    where each value is a list of (text: str, labels: list[int]) tuples.
    """
    splits = {"train": [], "validation": [], "test": []}
    split_keys = list(splits.keys())

    for ds_name, ds_id in DATASET_IDS.items():
        result = _load_one_dataset(ds_id, lang)
        if result is None:
            continue
        for i, key in enumerate(split_keys):
            splits[key].extend(result[i])

    # Deduplicate each split by normalized text
    for key in split_keys:
        seen = set()
        deduped = []
        for text, labels in splits[key]:
            if text not in seen:
                seen.add(text)
                deduped.append((text, labels))
        splits[key] = deduped
        logger.info(f"Split '{key}': {len(deduped)} examples after dedup")

    return splits


def compute_pos_weights(train_labels: np.ndarray) -> torch.Tensor:
    """Per-class pos_weight = n_neg / n_pos for BCEWithLogitsLoss."""
    n = len(train_labels)
    pos = train_labels.sum(axis=0).clip(min=1)
    neg = n - pos
    return torch.tensor(neg / pos, dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics. Frozen after Stage 1 — do not modify."""
    num_labels = len(EMOTIONS)
    label_idx = list(range(num_labels))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=label_idx, zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    jaccard = float(jaccard_score(y_true, y_pred, average="macro", zero_division=0))

    per_class = {
        EMOTIONS[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i in range(num_labels)
    }

    return {"macro_f1": macro_f1, "jaccard": jaccard, "per_class": per_class}


def majority_label_baseline(train_labels: np.ndarray, test_labels: np.ndarray) -> dict:
    """Predict the most frequent label combination from train for every test example."""
    counter = Counter(tuple(row) for row in train_labels)
    most_common = np.array(counter.most_common(1)[0][0])
    preds = np.tile(most_common, (len(test_labels), 1))
    return compute_metrics(test_labels, preds)
