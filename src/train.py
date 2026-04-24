"""
Stage 1: Fine-tuning pipeline for multilabel emotion classification.
Usage: python train.py --model xlm-roberta --lang sw
"""

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    EMOTIONS,
    TEXT_COL,
    compute_metrics,
    compute_pos_weights,
    load_language_data,
    majority_label_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
    "afro-xlmr": "Davlan/afro-xlmr-base",
}


class EmotionDataset(Dataset):
    def __init__(self, texts: list, labels: np.ndarray, tokenizer, max_length: int):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class EmotionClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = len(EMOTIONS)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    Run one epoch. Pass optimizer for training; omit for evaluation.
    Returns (avg_loss, predictions_array, labels_array).
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = nullcontext() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].int().numpy())

    return (
        total_loss / len(loader),
        np.vstack(all_preds),
        np.vstack(all_labels),
    )


def main():
    parser = argparse.ArgumentParser(description="Train multilabel emotion classifier")
    parser.add_argument(
        "--model",
        choices=list(SUPPORTED_MODELS.keys()),
        default="xlm-roberta",
        help="Model architecture to fine-tune",
    )
    parser.add_argument("--lang", required=True, help="Language code, e.g. sw, amh, afr, yo")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience on val macro-F1")
    parser.add_argument("--data_dir", default=None, help="Load augmented dataset from local path (output of augment.py)")
    args = parser.parse_args()

    model_name = SUPPORTED_MODELS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model: {model_name} | Lang: {args.lang} | Device: {device}")

    # ── Data loading ─────────────────────────────────────────────────────────
    if args.data_dir:
        from datasets import load_from_disk
        raw = load_from_disk(args.data_dir)
        splits = {}
        for split_name, hf_split in raw.items():
            rows = []
            for example in hf_split:
                text = example[TEXT_COL]
                labels = [int(example[e]) for e in EMOTIONS]
                rows.append((text, labels))
            splits[split_name] = rows
        # Normalise split key: DatasetDict from augment.py uses "validation"
        if "validation" not in splits and "dev" in splits:
            splits["validation"] = splits.pop("dev")
    else:
        splits = load_language_data(args.lang)

    for split_name in ("train", "validation", "test"):
        if not splits[split_name]:
            raise ValueError(f"No '{split_name}' data found for lang={args.lang}")

    def unzip(pairs):
        texts, labels = zip(*pairs)
        return list(texts), np.array(labels, dtype=np.float32)

    train_texts, train_labels = unzip(splits["train"])
    val_texts, val_labels = unzip(splits["validation"])
    test_texts, test_labels = unzip(splits["test"])

    # ── Majority-label baseline (computed before training) ────────────────────
    baseline = majority_label_baseline(train_labels.astype(int), test_labels.astype(int))
    logger.info(f"Majority-label baseline macro-F1: {baseline['macro_f1']:.4f}")

    # ── Tokenisation ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = EmotionDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_ds = EmotionDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_ds = EmotionDataset(test_texts, test_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model = EmotionClassifier(model_name).to(device)
    pos_weights = compute_pos_weights(train_labels.astype(int)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # ── Training loop with early stopping ─────────────────────────────────────
    checkpoint_dir = Path("models") / args.model / args.lang
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        _, val_preds, val_true = run_epoch(model, val_loader, criterion, device)

        val_metrics = compute_metrics(val_true, val_preds)
        val_f1 = val_metrics["macro_f1"]
        logger.info(f"Epoch {epoch}/{args.epochs} — train_loss={train_loss:.4f}, val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  → Best checkpoint saved (val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  → No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    _, test_preds, test_true = run_epoch(model, test_loader, criterion, device)

    test_metrics = compute_metrics(test_true, test_preds)
    test_metrics["majority_label_baseline"] = baseline

    results_path = Path("results") / args.model / args.lang / "metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"Test macro-F1:  {test_metrics['macro_f1']:.4f}")
    logger.info(f"Majority baseline: {baseline['macro_f1']:.4f}")
    logger.info(f"Results saved → {results_path}")

    if test_metrics["macro_f1"] > baseline["macro_f1"]:
        logger.info("SUCCESS: model exceeds majority-label baseline")
    else:
        logger.warning("Model does NOT exceed majority-label baseline")


if __name__ == "__main__":
    main()
