"""
Stage 3: Explainability & Attention Visualization for multilabel emotion classification.

Languages: Afrikaans (afr), Xhosa (xho), Swahili (swa), Amharic (amh), Zulu (zul)
Models: mBERT, XLM-RoBERTa, AfroXLMR

Usage:
  python src/explain.py --model xlm-roberta --lang amh --text "I am angry"
  python src/explain.py --model mbert --lang afr --visualize_type head_view
  python src/explain.py --compare_langs amh swa xho --model xlm-roberta
  python src/explain.py --model xlm-roberta --lang swa --identify_cues
"""

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

# Supported languages for the project
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
    """Same architecture as train.py for compatibility."""
    def __init__(self, model_name: str, num_labels: int = len(EMOTIONS)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        return logits, outputs.attentions, outputs.last_hidden_state


def load_trained_model(model_name: str, lang: str, device: str = "cpu"):
    """
    Load a trained model checkpoint from models/{model_name}/{lang}/best_model.pt
    Returns (model, tokenizer, config) or None if not found.
    """
    checkpoint_path = Path("models") / model_name / lang / "best_model.pt"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Available checkpoints should be in: models/{model_name}/{lang}/")
        return None, None, None

    try:
        model = EmotionClassifier(SUPPORTED_MODELS[model_name])
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(SUPPORTED_MODELS[model_name])

        # Load config if available
        config_path = checkpoint_path.parent / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        logger.info(f"Loaded model from {checkpoint_path}")
        return model, tokenizer, config

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None, None


def predict_emotions(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
    threshold: float = 0.5
) -> Tuple[Dict, np.ndarray]:
    """
    Predict emotions for a text and return predictions + probabilities.
    Returns (predictions_dict, logits).
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits, attentions, hidden_states = model(input_ids, attention_mask)

    # Convert logits to probabilities
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Threshold predictions
    predictions = {
        emotion: {
            "probability": float(probs[i]),
            "predicted": bool(probs[i] > threshold)
        }
        for i, emotion in enumerate(EMOTIONS)
    }

    return predictions, probs, attentions, hidden_states, inputs


def extract_token_importance(
    attentions: Tuple,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    emotion_idx: int = None,
    method: str = "gradient"
) -> Dict:
    """
    Extract token importance from attention weights.
    Methods:
      - "attention_rollout": Compute cumulative attention across layers
      - "attention_head": Average attention heads
      - "last_layer": Use only last layer attention

    Returns dict: {token: importance_score}
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    if method == "attention_rollout":
        # Rollout: combine attention across layers
        rollout = torch.eye(len(tokens), device=attentions[0].device)
        for attention in attentions:
            # attention shape: (batch_size, num_heads, seq_len, seq_len)
            # Average over heads
            attention_heads_fused = attention.mean(dim=1)[0]
            rollout = torch.matmul(attention_heads_fused, rollout)

        # Take [CLS] token (index 0) as reference
        importance = rollout[0, :].cpu().numpy()

    elif method == "last_layer":
        # Use attention from last layer, averaged over heads
        last_attention = attentions[-1][0]  # (num_heads, seq_len, seq_len)
        importance = last_attention.mean(dim=0)[0, :].cpu().numpy()  # CLS token attention

    else:  # "attention_head" (default)
        # Simple: average attention over all layers and heads
        all_attentions = []
        for layer_attention in attentions:
            layer_attention = layer_attention[0]  # Remove batch dim
            all_attentions.append(layer_attention.mean(dim=0)[0, :])  # CLS attention

        importance = torch.stack(all_attentions).mean(dim=0).cpu().numpy()

    # Normalize to [0, 1]
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    token_importance = {
        token: float(score)
        for token, score in zip(tokens, importance)
    }

    return token_importance


def visualize_attention(
    attentions: Tuple,
    tokens: List[str],
    model_name: str,
    lang: str,
    output_dir: str = "outputs",
    visualize_type: str = "head_view"
) -> str:
    """
    Visualize attention weights using BERTViz.
    visualize_type: "head_view" (per-head) or "model_view" (averaged)
    Returns path to saved HTML file.
    """
    if not BERTVIZ_AVAILABLE:
        logger.warning("BERTViz not available. Install with: pip install bertviz")
        return None

    Path(output_dir).mkdir(exist_ok=True)

    # Prepare attention tensors: BERTViz expects (batch_size=1, num_heads, seq_len, seq_len)
    attention_tensors = list(attentions)

    output_file = Path(output_dir) / f"{model_name}_{lang}_attention.html"

    try:
        # We need to capture the HTML string correctly from BERTViz
        if visualize_type == "head_view":
            html = head_view(attention_tensors, tokens, html_action='return')
        else:  # model_view
            html = model_view(attention_tensors, tokens, html_action='return')

        # BERTViz returns an IPython.display.HTML object when html_action='return'
        # We can extract the HTML string from its .data attribute
        with open(output_file, "w") as f:
            f.write(html.data)

        logger.info(f"Attention visualization saved to {output_file}")
        return str(output_file)

    except Exception as e:
        logger.error(f"Failed to visualize attention: {e}")
        return None


def compare_attention_across_languages(
    model_name: str,
    languages: List[str],
    text: str,
    device: str = "cpu"
) -> Dict:
    """
    Compare attention patterns across multiple languages for the same text.
    Useful for cross-lingual analysis.

    Returns: {lang: {emotion: {token: importance}}}
    """
    results = {}

    for lang in languages:
        logger.info(f"\nAnalyzing {model_name} on {lang}...")

        model, tokenizer, _ = load_trained_model(model_name, lang, device)
        if model is None:
            logger.warning(f"Skipping {lang} (model not found)")
            continue

        predictions, probs, attentions, _, inputs = predict_emotions(
            model, tokenizer, text, device
        )

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Extract importance for each emotion
        lang_results = {}
        for emotion_idx, emotion in enumerate(EMOTIONS):
            importance = extract_token_importance(
                attentions,
                inputs["input_ids"],
                tokenizer,
                emotion_idx=emotion_idx,
                method="attention_rollout"
            )
            lang_results[emotion] = importance

        results[lang] = {
            "predictions": predictions,
            "token_importance": lang_results,
            "tokens": tokens
        }

    return results


def identify_culturally_specific_cues(
    model_name: str,
    lang: str,
    dataset_split: str = "test",
    top_k: int = 10,
    device: str = "cpu"
) -> Dict:
    """
    Analyze which tokens are most important for each emotion.
    Useful for identifying culturally specific emotional expressions.

    Returns: {emotion: [(token, avg_importance, frequency)]}
    """
    logger.info(f"Loading {lang} dataset...")

    try:
        data_splits = load_language_data(lang)
        # Fall back to "test" split if the requested split is unavailable
        test_data = data_splits.get(dataset_split, data_splits.get("test", []))
    except Exception as e:
        logger.error(f"Failed to load data for {lang}: {e}")
        return {}

    model, tokenizer, _ = load_trained_model(model_name, lang, device)
    if model is None:
        return {}

    # Aggregate token importance per emotion
    emotion_tokens = {emotion: [] for emotion in EMOTIONS}
    token_frequencies = {emotion: {} for emotion in EMOTIONS}

    # Sample up to 500 texts for efficiency
    sample_data = test_data[:500]
    sample_texts = [row[0] for row in sample_data]

    logger.info(f"Processing {len(sample_texts)} texts...")

    for idx, text in enumerate(sample_texts):
        if idx % 100 == 0:
            logger.info(f"  Processed {idx}/{len(sample_texts)}")

        try:
            predictions, _, attentions, _, inputs = predict_emotions(
                model, tokenizer, text, device
            )

            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            for emotion_idx, emotion in enumerate(EMOTIONS):
                # Check if the text is labeled with this emotion (1 for present)
                if sample_data[idx][1][emotion_idx] == 1:
                    importance = extract_token_importance(
                        attentions,
                        inputs["input_ids"],
                        tokenizer,
                        emotion_idx=emotion_idx
                    )

                    for token, score in importance.items():
                        emotion_tokens[emotion].append((token, score))
                        token_frequencies[emotion][token] = token_frequencies[emotion].get(token, 0) + 1

        except Exception as e:
            logger.debug(f"Error processing text {idx}: {e}")
            continue

    # Aggregate results
    results = {}
    for emotion in EMOTIONS:
        if emotion_tokens[emotion]:
            # Group by token and average importance
            token_scores = {}
            for token, score in emotion_tokens[emotion]:
                if token not in token_scores:
                    token_scores[token] = []
                token_scores[token].append(score)

            # Calculate mean importance and frequency
            token_stats = [
                (token, np.mean(scores), token_frequencies[emotion].get(token, 0))
                for token, scores in token_scores.items()
            ]

            # Sort by importance and take top-k
            token_stats.sort(key=lambda x: x[1], reverse=True)
            results[emotion] = token_stats[:top_k]
        else:
            results[emotion] = []

    return results


def save_analysis_report(
    analysis_results: Dict,
    model_name: str,
    lang: str,
    output_dir: str = "outputs"
) -> str:
    """Save analysis results to JSON for later inspection."""
    Path(output_dir).mkdir(exist_ok=True)

    report_path = Path(output_dir) / f"{model_name}_{lang}_analysis.json"

    # Convert numpy types to native Python types for JSON serialization
    serializable_results = {}
    for emotion, tokens in analysis_results.items():
        serializable_results[emotion] = [
            (token, float(score), int(freq))
            for token, score, freq in tokens
        ]

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
        help="Model to analyze (mbert, xlm-roberta, afro-xlmr)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Language code: afr (Afrikaans), xho (Xhosa), swa (Swahili), amh (Amharic), zul (Zulu)"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input text to analyze"
    )
    parser.add_argument(
        "--visualize_type",
        type=str,
        choices=["head_view", "model_view"],
        default="head_view",
        help="Attention visualization type"
    )
    parser.add_argument(
        "--compare_langs",
        nargs="+",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Compare attention across languages (e.g., amh swa xho afr zul)"
    )
    parser.add_argument(
        "--identify_cues",
        action="store_true",
        help="Identify culturally specific emotional cues"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for visualizations and reports"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Single text analysis
    if args.text and args.lang:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing text: '{args.text}'")
        logger.info(f"Model: {args.model}, Language: {args.lang}")
        logger.info(f"{'='*60}\n")

        model, tokenizer, config = load_trained_model(args.model, args.lang, args.device)
        if model is None:
            logger.error("Failed to load model. Exiting.")
            return

        predictions, probs, attentions, hidden_states, inputs = predict_emotions(
            model, tokenizer, args.text, args.device
        )

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Print predictions
        logger.info("Emotion Predictions:")
        logger.info("-" * 40)
        for emotion, pred in predictions.items():
            prob = pred["probability"]
            status = "✓" if pred["predicted"] else "✗"
            logger.info(f"  {status} {emotion:12} {prob:.4f}")

        # Extract and print token importance
        token_importance = extract_token_importance(
            attentions,
            inputs["input_ids"],
            tokenizer,
            method="attention_rollout"
        )

        logger.info("\nToken Importance (Top-10):")
        logger.info("-" * 40)
        sorted_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)
        for token, score in sorted_tokens[:10]:
            logger.info(f"  {token:15} {score:.4f}")

        # Visualize attention
        if BERTVIZ_AVAILABLE:
            viz_path = visualize_attention(
                attentions,
                tokens,
                args.model,
                args.lang,
                args.output_dir,
                args.visualize_type
            )
            if viz_path:
                logger.info(f"\nVisualization saved: {viz_path}")

    # Cross-lingual comparison
    elif args.compare_langs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-lingual Comparison")
        logger.info(f"Languages: {', '.join(args.compare_langs)}")
        logger.info(f"Model: {args.model}")
        logger.info(f"{'='*60}\n")

        if not args.text:
            args.text = "I am very angry and sad"
            logger.info(f"Using default text: '{args.text}'\n")

        comparison = compare_attention_across_languages(
            args.model,
            args.compare_langs,
            args.text,
            args.device
        )

        # Print cross-lingual results
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
            tokens = comparison[lang]["tokens"]
            emotions = comparison[lang]["token_importance"]

            # Show top tokens for predicted emotions
            for emotion, importance_dict in emotions.items():
                preds = comparison[lang]["predictions"]
                if preds[emotion]["predicted"]:
                    sorted_tokens = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    top_tokens = [t for t, _ in sorted_tokens[:5]]
                    logger.info(f"  {emotion}: {', '.join(top_tokens)}")

    # Identify culturally specific cues
    elif args.identify_cues and args.lang:
        logger.info(f"\n{'='*60}")
        logger.info(f"Identifying Culturally-Specific Emotional Cues")
        logger.info(f"Model: {args.model}, Language: {args.lang}")
        logger.info(f"{'='*60}\n")

        cues = identify_culturally_specific_cues(
            args.model,
            args.lang,
            device=args.device
        )

        # Print and save results
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
