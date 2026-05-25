import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

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
    "afroberta": "castorini/afriberta_large",
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

def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    # one lap around the training track
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].int().numpy())

    return (
        total_loss / len(loader),
        np.vstack(all_preds),
        np.vstack(all_labels),
    )

def tune_thresholds(model, loader, device):
    # finding the sweet spot for each emotion's sensitivity
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch["labels"].int())

    val_probs = torch.cat(all_logits).numpy()
    val_true = torch.cat(all_labels).numpy()

    best_thresholds = []
    logger.info("Per-label threshold tuning on validation set:")
    for i, emotion in enumerate(EMOTIONS):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.025):
            preds = (val_probs[:, i] >= t).astype(int)
            f1 = f1_score(val_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds.append(round(float(best_t), 3))
        logger.info(f"  {emotion}: best_threshold={best_t:.3f}, val_f1={best_f1:.4f}")

    return np.array(best_thresholds)

def evaluate_with_thresholds(model, loader, device, thresholds):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= thresholds).astype(int)
            all_preds.append(preds)
            all_labels.append(batch["labels"].int().numpy())

    return np.vstack(all_preds), np.vstack(all_labels)

def train_and_evaluate(
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
    model_name, args, device, checkpoint_dir, fold_label="",
):
    prefix = f"[{fold_label}] " if fold_label else ""

    baseline = majority_label_baseline(train_labels.astype(int), test_labels.astype(int))
    logger.info(f"{prefix}Majority-label baseline macro-F1: {baseline['macro_f1']:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = EmotionDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_ds = EmotionDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_ds = EmotionDataset(test_texts, test_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = EmotionClassifier(model_name).to(device)
    pos_weights = compute_pos_weights(train_labels.astype(int)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"{prefix}Scheduler: cosine w/ warmup — {warmup_steps}/{total_steps} warmup steps")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scheduler=scheduler,
        )
        _, val_preds, val_true = run_epoch(model, val_loader, criterion, device)

        val_metrics = compute_metrics(val_true, val_preds)
        val_f1 = val_metrics["macro_f1"]
        logger.info(f"{prefix}Epoch {epoch}/{args.epochs} — train_loss={train_loss:.4f}, val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"{prefix}  → Best checkpoint saved (val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"{prefix}  → No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info(f"{prefix}Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    thresholds = tune_thresholds(model, val_loader, device)

    test_preds, test_true = evaluate_with_thresholds(model, test_loader, device, thresholds)
    test_metrics = compute_metrics(test_true, test_preds)
    test_metrics["majority_label_baseline"] = baseline

    logger.info(f"{prefix}Test macro-F1:  {test_metrics['macro_f1']:.4f}")
    logger.info(f"{prefix}Majority baseline: {baseline['macro_f1']:.4f}")

    if test_metrics["macro_f1"] > baseline["macro_f1"]:
        logger.info(f"{prefix}SUCCESS: model exceeds majority-label baseline")
    else:
        logger.warning(f"{prefix}Model does NOT exceed majority-label baseline")

    return test_metrics, thresholds.tolist(), baseline

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
    parser.add_argument("--gpu_limit", type=float, default=0.8, help="Limit GPU memory usage fraction (0.0 to 1.0)")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for results and models directories (e.g. '_combined')")
    parser.add_argument("--output_dir", type=str, default=".", help="Base directory to save results and models")
    parser.add_argument("--cv_folds", type=int, default=1, help="Number of cross-validation folds (>1 enables CV mode)")
    parser.add_argument("--cv_min_val_size", type=int, default=150, help="Val set size threshold below which CV is recommended")
    args = parser.parse_args()

    model_name = SUPPORTED_MODELS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Model: {model_name} | Lang: {args.lang} | Device: {device}")

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

    if len(val_texts) < args.cv_min_val_size and args.cv_folds <= 1:
        logger.warning(
            f"Validation set has only {len(val_texts)} samples (< {args.cv_min_val_size}). "
            f"Consider using --cv_folds 3 for more stable evaluation."
        )

    models_dir = Path(args.output_dir) / f"models{args.output_suffix}"
    results_dir = Path(args.output_dir) / f"results{args.output_suffix}"

    if args.cv_folds > 1:
        logger.info(f"Cross-validation mode: {args.cv_folds} folds")

        all_cv_texts = train_texts + val_texts
        all_cv_labels = np.vstack([train_labels, val_labels])
        n = len(all_cv_texts)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        fold_size = n // args.cv_folds
        fold_metrics = []

        for fold_idx in range(args.cv_folds):
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold_idx + 1}/{args.cv_folds}")
            logger.info(f"{'='*60}")

            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < args.cv_folds - 1 else n
            fold_val_idx = indices[val_start:val_end]
            fold_train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            fold_train_texts = [all_cv_texts[i] for i in fold_train_idx]
            fold_train_labels = all_cv_labels[fold_train_idx]
            fold_val_texts = [all_cv_texts[i] for i in fold_val_idx]
            fold_val_labels = all_cv_labels[fold_val_idx]

            logger.info(f"  Train: {len(fold_train_texts)}, Val: {len(fold_val_texts)}, Test: {len(test_texts)}")

            fold_checkpoint_dir = Path(models_dir) / args.model / args.lang / f"fold_{fold_idx + 1}"

            metrics, thresholds, baseline = train_and_evaluate(
                fold_train_texts, fold_train_labels,
                fold_val_texts, fold_val_labels,
                test_texts, test_labels,
                model_name, args, device, fold_checkpoint_dir,
                fold_label=f"Fold {fold_idx + 1}",
            )
            fold_metrics.append(metrics)

            fold_results_path = Path(results_dir) / args.model / args.lang / f"fold_{fold_idx + 1}" / "metrics.json"
            fold_results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fold_results_path, "w") as f:
                json.dump(metrics, f, indent=2)

            fold_thresholds_path = fold_results_path.parent / "thresholds.json"
            with open(fold_thresholds_path, "w") as f:
                json.dump(dict(zip(EMOTIONS, thresholds)), f, indent=2)

        macro_f1s = [m["macro_f1"] for m in fold_metrics]
        mean_f1 = float(np.mean(macro_f1s))
        std_f1 = float(np.std(macro_f1s))

        logger.info(f"\n{'='*60}")
        logger.info(f"CV RESULTS ({args.cv_folds} folds)")
        logger.info(f"  Mean test macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"  Per-fold: {[f'{f1:.4f}' for f1 in macro_f1s]}")
        logger.info(f"{'='*60}")

        cv_results = {
            "cv_folds": args.cv_folds,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "per_fold_macro_f1": macro_f1s,
            "per_fold_metrics": fold_metrics,
        }
        cv_path = Path(results_dir) / args.model / args.lang / "cv_results.json"
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cv_path, "w") as f:
            json.dump(cv_results, f, indent=2)
        logger.info(f"CV results saved → {cv_path}")

    else:
        checkpoint_dir = Path(models_dir) / args.model / args.lang

        test_metrics, thresholds, baseline = train_and_evaluate(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels,
            model_name, args, device, checkpoint_dir,
        )

        results_path = Path(results_dir) / args.model / args.lang / "metrics.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

        thresholds_path = results_path.parent / "thresholds.json"
        with open(thresholds_path, "w") as f:
            json.dump(dict(zip(EMOTIONS, thresholds)), f, indent=2)

        logger.info(f"Results saved → {results_path}")
        logger.info(f"Thresholds saved → {thresholds_path}")

if __name__ == "__main__":
    main()
