import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import EMOTIONS, TEXT_COL, load_language_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
    "afro-xlmr": "Davlan/afro-xlmr-base",
}

SUPPORTED_LANGUAGES = {
    "afr": "Afrikaans",
    "xho": "Xhosa",
    "swa": "Swahili",
    "amh": "Amharic",
    "zul": "Zulu",
}

try:
    from bertviz import head_view, model_view

    BERTVIZ_AVAILABLE = True
except ImportError:
    logger.warning("BERTViz not installed. Install with: pip install bertviz")
    BERTVIZ_AVAILABLE = False


class EmotionClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = len(EMOTIONS)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
        )
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        return logits, outputs.attentions, outputs.last_hidden_state


def load_trained_model(
    model_name: str, lang: str, device: str = "cpu", models_dir: str = "models"
):
    checkpoint_path = Path(models_dir) / model_name / lang / "best_model.pt"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None, None, None

    try:
        model = EmotionClassifier(SUPPORTED_MODELS[model_name])
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(SUPPORTED_MODELS[model_name])

        thresholds = np.full(len(EMOTIONS), 0.5)
        metrics_path = Path("results") / model_name / lang / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                saved = json.load(f)
            if "thresholds" in saved:
                thresholds = np.array(
                    [saved["thresholds"].get(e, 0.5) for e in EMOTIONS]
                )
                logger.info(f"Loaded per-class thresholds from {metrics_path}")

        logger.info(f"Loaded model from {checkpoint_path}")
        return model, tokenizer, thresholds

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None, None


def predict_emotions(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
    thresholds: np.ndarray = None,
) -> Tuple[Dict, np.ndarray, Tuple, torch.Tensor, Dict]:
    if thresholds is None:
        thresholds = np.full(len(EMOTIONS), 0.5)

    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits, attentions, hidden_states = model(input_ids, attention_mask)

    probs = torch.sigmoid(logits).cpu().numpy()[0]

    predictions = {
        emotion: {
            "probability": float(probs[i]),
            "threshold": float(thresholds[i]),
            "predicted": bool(probs[i] >= thresholds[i]),
        }
        for i, emotion in enumerate(EMOTIONS)
    }

    return predictions, probs, attentions, hidden_states, inputs


def extract_token_importance(
    attentions: Tuple,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    emotion_idx: int = None,
    method: str = "gradient",
) -> Dict:
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    if method == "attention_rollout":
        rollout = torch.eye(len(tokens), device=attentions[0].device)
        for attention in attentions:
            attention_heads_fused = attention.mean(dim=1)[0]
            rollout = torch.matmul(attention_heads_fused, rollout)
        importance = rollout[0, :].cpu().numpy()

    elif method == "last_layer":
        last_attention = attentions[-1][0]
        importance = last_attention.mean(dim=0)[0, :].cpu().numpy()

    else:
        all_attentions = []
        for layer_attention in attentions:
            layer_attention = layer_attention[0]
            all_attentions.append(layer_attention.mean(dim=0)[0, :])
        importance = torch.stack(all_attentions).mean(dim=0).cpu().numpy()

    importance = (importance - importance.min()) / (
        importance.max() - importance.min() + 1e-8
    )

    return {token: float(score) for token, score in zip(tokens, importance)}


def visualize_attention(
    attentions: Tuple,
    tokens: List[str],
    model_name: str,
    lang: str,
    output_dir: str = "outputs",
    visualize_type: str = "head_view",
) -> str:
    if not BERTVIZ_AVAILABLE:
        logger.warning("BERTViz not available. Install with: pip install bertviz")
        return None

    Path(output_dir).mkdir(exist_ok=True)

    attention_tensors = list(attentions)
    output_file = Path(output_dir) / f"{model_name}_{lang}_attention.html"

    try:
        if visualize_type == "head_view":
            html = head_view(attention_tensors, tokens, html_action="return")
        else:
            html = model_view(attention_tensors, tokens, html_action="return")

        with open(output_file, "w") as f:
            f.write(html.data)

        logger.info(f"Attention visualization saved to {output_file}")
        return str(output_file)

    except Exception as e:
        logger.error(f"Failed to visualize attention: {e}")
        return None


def compare_attention_across_languages(
    model_name: str, languages: List[str], text: str, device: str = "cpu"
) -> Dict:
    results = {}

    for lang in languages:
        logger.info(f"\nAnalyzing {model_name} on {lang}...")

        model, tokenizer, thresholds = load_trained_model(model_name, lang, device)
        if model is None:
            logger.warning(f"Skipping {lang} (model not found)")
            continue

        predictions, probs, attentions, _, inputs = predict_emotions(
            model, tokenizer, text, device, thresholds
        )

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        lang_results = {}
        for emotion_idx, emotion in enumerate(EMOTIONS):
            importance = extract_token_importance(
                attentions,
                inputs["input_ids"],
                tokenizer,
                emotion_idx=emotion_idx,
                method="attention_rollout",
            )
            lang_results[emotion] = importance

        results[lang] = {
            "predictions": predictions,
            "token_importance": lang_results,
            "tokens": tokens,
        }

    return results


def identify_culturally_specific_cues(
    model_name: str,
    lang: str,
    dataset_split: str = "test",
    top_k: int = 10,
    device: str = "cpu",
) -> Dict:
    logger.info(f"Loading {lang} dataset...")

    try:
        data_splits = load_language_data(lang)
        test_data = data_splits.get(dataset_split, data_splits.get("test", []))
    except Exception as e:
        logger.error(f"Failed to load data for {lang}: {e}")
        return {}

    model, tokenizer, thresholds = load_trained_model(model_name, lang, device)
    if model is None:
        return {}

    emotion_tokens = {emotion: [] for emotion in EMOTIONS}
    token_frequencies = {emotion: {} for emotion in EMOTIONS}

    sample_data = test_data[:500]
    sample_texts = [row[0] for row in sample_data]

    logger.info(f"Processing {len(sample_texts)} texts...")

    for idx, text in enumerate(sample_texts):
        if idx % 100 == 0:
            logger.info(f"  Processed {idx}/{len(sample_texts)}")

        try:
            predictions, _, attentions, _, inputs = predict_emotions(
                model, tokenizer, text, device, thresholds
            )

            for emotion_idx, emotion in enumerate(EMOTIONS):
                if sample_data[idx][1][emotion_idx] == 1:
                    importance = extract_token_importance(
                        attentions,
                        inputs["input_ids"],
                        tokenizer,
                        emotion_idx=emotion_idx,
                    )

                    for token, score in importance.items():
                        emotion_tokens[emotion].append((token, score))
                        token_frequencies[emotion][token] = (
                            token_frequencies[emotion].get(token, 0) + 1
                        )

        except Exception as e:
            logger.debug(f"Error processing text {idx}: {e}")
            continue

    results = {}
    for emotion in EMOTIONS:
        if emotion_tokens[emotion]:
            token_scores = {}
            for token, score in emotion_tokens[emotion]:
                if token not in token_scores:
                    token_scores[token] = []
                token_scores[token].append(score)

            token_stats = [
                (token, np.mean(scores), token_frequencies[emotion].get(token, 0))
                for token, scores in token_scores.items()
            ]

            token_stats.sort(key=lambda x: x[1], reverse=True)
            results[emotion] = token_stats[:top_k]
        else:
            results[emotion] = []

    return results


def save_analysis_report(
    analysis_results: Dict, model_name: str, lang: str, output_dir: str = "outputs"
) -> str:
    Path(output_dir).mkdir(exist_ok=True)

    report_path = Path(output_dir) / f"{model_name}_{lang}_analysis.json"

    serializable_results = {
        emotion: [(token, float(score), int(freq)) for token, score, freq in tokens]
        for emotion, tokens in analysis_results.items()
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Analysis report saved to {report_path}")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="Explainability analysis for emotion classification models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(SUPPORTED_MODELS.keys()),
        default="xlm-roberta",
    )
    parser.add_argument("--lang", type=str, choices=list(SUPPORTED_LANGUAGES.keys()))
    parser.add_argument("--text", type=str)
    parser.add_argument(
        "--visualize_type",
        type=str,
        choices=["head_view", "model_view"],
        default="head_view",
    )
    parser.add_argument(
        "--compare_langs", nargs="+", choices=list(SUPPORTED_LANGUAGES.keys())
    )
    parser.add_argument("--identify_cues", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    if args.text and args.lang:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Analyzing text: '{args.text}'")
        logger.info(f"Model: {args.model}, Language: {args.lang}")
        logger.info(f"{'=' * 60}\n")

        model, tokenizer, thresholds = load_trained_model(
            args.model, args.lang, args.device
        )
        if model is None:
            logger.error("Failed to load model. Exiting.")
            return

        predictions, probs, attentions, hidden_states, inputs = predict_emotions(
            model, tokenizer, args.text, args.device, thresholds
        )

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        logger.info("Emotion Predictions (prob / threshold):")
        logger.info("-" * 40)
        for emotion, pred in predictions.items():
            status = "✓" if pred["predicted"] else "✗"
            logger.info(
                f"  {status} {emotion:12} {pred['probability']:.4f}  (threshold={pred['threshold']:.2f})"
            )

        token_importance = extract_token_importance(
            attentions, inputs["input_ids"], tokenizer, method="attention_rollout"
        )

        logger.info("\nToken Importance (Top-5):")
        logger.info("-" * 40)
        for token, score in sorted(
            token_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            logger.info(f"  {token:15} {score:.4f}")

        if BERTVIZ_AVAILABLE:
            viz_path = visualize_attention(
                attentions,
                tokens,
                args.model,
                args.lang,
                args.output_dir,
                args.visualize_type,
            )
            if viz_path:
                logger.info(f"\nVisualization saved: {viz_path}")

    elif args.compare_langs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Cross-lingual Comparison")
        logger.info(f"Languages: {', '.join(args.compare_langs)}")
        logger.info(f"Model: {args.model}")
        logger.info(f"{'=' * 60}\n")

        if not args.text:
            args.text = "I am very angry and sad"
            logger.info(f"Using default text: '{args.text}'\n")

        comparison = compare_attention_across_languages(
            args.model, args.compare_langs, args.text, args.device
        )

        for lang in args.compare_langs:
            if lang not in comparison:
                continue

            logger.info(f"\n{lang.upper()}:")
            logger.info("-" * 40)
            logger.info("Predictions:")
            for emotion, pred in comparison[lang]["predictions"].items():
                if pred["predicted"]:
                    logger.info(f"  ✓ {emotion}: {pred['probability']:.4f}")

            logger.info("Top tokens:")
            for emotion, importance_dict in comparison[lang][
                "token_importance"
            ].items():
                if comparison[lang]["predictions"][emotion]["predicted"]:
                    top_tokens = [
                        t
                        for t, _ in sorted(
                            importance_dict.items(), key=lambda x: x[1], reverse=True
                        )[:5]
                    ]
                    logger.info(f"  {emotion}: {', '.join(top_tokens)}")

    elif args.identify_cues and args.lang:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Identifying Culturally-Specific Emotional Cues")
        logger.info(f"Model: {args.model}, Language: {args.lang}")
        logger.info(f"{'=' * 60}\n")

        cues = identify_culturally_specific_cues(
            args.model, args.lang, device=args.device
        )

        logger.info("\nTop Tokens per Emotion:")
        logger.info("-" * 60)
        for emotion, tokens in cues.items():
            if tokens:
                logger.info(f"\n{emotion.upper()}:")
                for token, score, freq in tokens[:5]:
                    logger.info(f"  {token:15} importance={score:.4f}, freq={freq}")

        save_analysis_report(cues, args.model, args.lang, args.output_dir)

    else:
        logger.error(
            "Please provide either:\n"
            "  1. --text and --lang (for single text analysis)\n"
            "  2. --compare_langs (for cross-lingual comparison)\n"
            "  3. --identify_cues and --lang (for culturally-specific cues)"
        )


if __name__ == "__main__":
    main()
