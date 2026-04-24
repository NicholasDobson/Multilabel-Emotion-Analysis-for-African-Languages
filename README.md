# Multilabel Emotion Analysis for African Languages  
**Transfer Learning + Data Augmentation on BRIGHTER & EthioEmo**

**Authors:** Nicholas Dobson, Simon van der Merwe, Naazneen Khan  
**Course:** Natural Language Processing COS 760 – University of Pretoria (2026)

---

## Overview
Natural Language Processing has largely focused on high-resource languages, leaving African languages underrepresented despite their vast diversity.

This project tackles **multilabel emotion classification** for African languages using:
- **BRIGHTER dataset** (28 languages)
- **EthioEmo dataset** (4 Ethiopian languages)

We explore how to improve performance in **low-resource settings** through:
- Transfer learning (multilingual transformers)
- Data augmentation (back-translation & paraphrasing)
- Explainability (attention visualization)

---

## Objectives
- Build strong baselines for African language emotion classification  
- Evaluate augmentation techniques for low-resource data  
- Analyse model behaviour using attention mechanisms  

---

## Research Questions
- **RQ1:** Which model performs best? (mBERT vs XLM-R vs AfroXLMR)  
- **RQ2:** How does data augmentation impact performance?  
- **RQ3:** What linguistic patterns do models learn across languages?  

---

## Datasets
- **BRIGHTER**: ~100k samples, 28 languages  
- **EthioEmo**: 23k samples, 4 languages  
- Optional: **AfriSenti** (auxiliary sentiment data)

### Emotion Labels (Multilabel)
- Anger  
- Fear  
- Surprise  
- Sadness  
- Happiness  
- Disgust  

---

## Methodology

### 1. Baseline Models
We fine-tune:
- mBERT  
- XLM-RoBERTa  
- AfroXLMR  

**Setup:**
- Loss: Binary Cross-Entropy (weighted)  
- LR: 2e-5  
- Batch size: 16  
- Epochs: 5 (early stopping on F1)

---

### 2. Data Augmentation
- **Back-translation** (via MarianMT / Google Translate)  
- **Paraphrasing** (multilingual T5)  

**Filtering:**
- Cosine similarity > 0.85 (semantic consistency)

---

### 3. Explainability
- Attention visualization using **BERTViz**
- Cross-lingual comparison of token importance
- Identification of culturally specific cues

---

## Evaluation Metrics
- **Macro F1 (primary)**
- Precision / Recall per label  
- Jaccard Similarity  
- Pearson correlation (optional intensity task)

### Baselines
- Majority classifier  
- mBERT (no augmentation)  
- SemEval-2025 benchmark systems  

---

## Expected Results
- AfroXLMR > XLM-R > mBERT for African languages  
- Data augmentation improves low-resource performance (~2–5% F1)  
- Errors cluster around:
  - Mixed emotions  
  - Culturally nuanced expressions  

---

## Responsible NLP Considerations
- Cultural variation in emotional expression  
- Annotator bias in datasets  
- Potential translation drift in augmentation  

**Mitigation:**
- Native-speaker annotated datasets  
- Similarity filtering  
- Error analysis across languages  

---

## Project Timeline
| Week | Task |
|------|------|
| 1–2 | Data preprocessing + baselines |
| 3–4 | Augmentation experiments |
| 5–6 | Explainability + write-up |

---

## Team Roles
- **Nicholas** – Modelling & Evaluation  
- **Simon** – Data Augmentation  
- **Naaz** – Explainability & Writing  

---

## Outputs
- Fine-tuned multilingual models  
- Augmentation pipeline  
- Attention-based analysis  
- Reproducible experiments  

---

## Setup & Usage

### Install dependencies
```bash
pip install -r requirements.txt
#or
python3 -m pip install -r requirements.txt
```


### Project structure
```
src/
  train.py      # Stage 1 — fine-tuning pipeline
  augment.py    # Stage 2 — data augmentation
  explain.py    # Stage 3 — attention visualisation (TODO)
  utils.py      # shared helpers (metrics, data loading, class weights)
data/           # downloaded and augmented datasets (gitignored)
models/         # saved checkpoints (gitignored)
results/        # metrics per model/language
outputs/        # attention visualisations (gitignored)
requirements.txt
```

### Stage 1 — Train a baseline model
```bash
# Fine-tune XLM-RoBERTa on Swahili Takes far too long 
python3 src/train.py --model xlm-roberta --lang swa 
# Smaller version
python3 src/train.py --model xlm-roberta --lang swh --epochs 2 --batch_size 8

# Fine-tune mBERT on Amharic
python3 src/train.py --model mbert --lang amh

# Fine-tune AfroXLMR with custom hyperparameters
python3 src/train.py --model afro-xlmr --lang yor --lr 1e-5 --batch_size 8 --epochs 10
```

**Available `--model` values:** `mbert`, `xlm-roberta`, `afro-xlmr`

Results are saved to `results/{model}/{lang}/metrics.json`.  
Best checkpoint is saved to `models/{model}/{lang}/best_model.pt`.

### Stage 2 — Augment training data
```bash
# Back-translate Amharic training data and save augmented dataset
python src/augment.py --lang amh --methods bt

# Train on augmented data
python src/train.py --model xlm-roberta --lang amh --data_dir data/amh_augmented
```

``` bash
# Check cache space from happyface 
du -sh ~/.cache/huggingface/hub/

# Delete ~3gb cache from running program
rm -rf ~/.cache/huggingface/hub/
```

---

## References
- BRIGHTER (ACL 2025)  
- EthioEmo (COLING 2025)  
- AfriSenti (EMNLP 2023)  
- SemEval-2025 Task 11  


---

## Future Work
- Extend to more African languages  
- Incorporate LLM-based prompting  
- Explore SHAP for deeper interpretability  
