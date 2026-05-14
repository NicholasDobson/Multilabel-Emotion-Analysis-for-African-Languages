# Complete Experiment Command List

This document contains EVERY single command you could possibly run for your Explainability experiments, broken down by Model and Language.

*Note: If you run a command and it says "Checkpoint not found", it just means you haven't run `train.py` for that specific model/language combination yet.*

---

## 1. Amharic (AMH) Experiments

### XLM-RoBERTa
**Single Sentence Analysis:**
```bash
python src/explain.py --model xlm-roberta --lang amh --text "በጣም አዝኛለሁ" --visualize_type head_view
open outputs/xlm-roberta_amh_attention.html
```

**Dataset Cultural Cues (500 sentences):**
```bash
python src/explain.py --model xlm-roberta --lang amh --identify_cues
open outputs/xlm-roberta_amh_analysis.json
```

### mBERT
**Single Sentence Analysis:**
```bash
python src/explain.py --model mbert --lang amh --text "በጣም አዝኛለሁ" --visualize_type head_view
open outputs/mbert_amh_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model mbert --lang amh --identify_cues
open outputs/mbert_amh_analysis.json
```

### AfroXLMR
**Single Sentence Analysis:**
```bash
python src/explain.py --model afro-xlmr --lang amh --text "በጣም አዝኛለሁ" --visualize_type head_view
open outputs/afro-xlmr_amh_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model afro-xlmr --lang amh --identify_cues
open outputs/afro-xlmr_amh_analysis.json
```

---

## 2. Swahili (SWA) Experiments

### XLM-RoBERTa
**Single Sentence Analysis:**
```bash
python src/explain.py --model xlm-roberta --lang swa --text "Nimekasirika sana" --visualize_type head_view
open outputs/xlm-roberta_swa_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model xlm-roberta --lang swa --identify_cues
open outputs/xlm-roberta_swa_analysis.json
```

### mBERT
**Single Sentence Analysis:**
```bash
python src/explain.py --model mbert --lang swa --text "Nimekasirika sana" --visualize_type head_view
open outputs/mbert_swa_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model mbert --lang swa --identify_cues
open outputs/mbert_swa_analysis.json
```

### AfroXLMR
**Single Sentence Analysis:**
```bash
python src/explain.py --model afro-xlmr --lang swa --text "Nimekasirika sana" --visualize_type head_view
open outputs/afro-xlmr_swa_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model afro-xlmr --lang swa --identify_cues
open outputs/afro-xlmr_swa_analysis.json
```

---

## 3. Xhosa (XHO) Experiments

### XLM-RoBERTa
**Single Sentence Analysis:**
```bash
python src/explain.py --model xlm-roberta --lang xho --text "Ndikhathazeke kakhulu" --visualize_type head_view
open outputs/xlm-roberta_xho_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model xlm-roberta --lang xho --identify_cues
open outputs/xlm-roberta_xho_analysis.json
```

### mBERT
**Single Sentence Analysis:**
```bash
python src/explain.py --model mbert --lang xho --text "Ndikhathazeke kakhulu" --visualize_type head_view
open outputs/mbert_xho_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model mbert --lang xho --identify_cues
open outputs/mbert_xho_analysis.json
```

### AfroXLMR
**Single Sentence Analysis:**
```bash
python src/explain.py --model afro-xlmr --lang xho --text "Ndikhathazeke kakhulu" --visualize_type head_view
open outputs/afro-xlmr_xho_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model afro-xlmr --lang xho --identify_cues
open outputs/afro-xlmr_xho_analysis.json
```

---

## 4. Afrikaans (AFR) Experiments

### XLM-RoBERTa
**Single Sentence Analysis:**
```bash
python src/explain.py --model xlm-roberta --lang afr --text "Ek is baie kwaad" --visualize_type head_view
open outputs/xlm-roberta_afr_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model xlm-roberta --lang afr --identify_cues
open outputs/xlm-roberta_afr_analysis.json
```

### mBERT
**Single Sentence Analysis:**
```bash
python src/explain.py --model mbert --lang afr --text "Ek is baie kwaad" --visualize_type head_view
open outputs/mbert_afr_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model mbert --lang afr --identify_cues
open outputs/mbert_afr_analysis.json
```

### AfroXLMR
**Single Sentence Analysis:**
```bash
python src/explain.py --model afro-xlmr --lang afr --text "Ek is baie kwaad" --visualize_type head_view
open outputs/afro-xlmr_afr_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model afro-xlmr --lang afr --identify_cues
open outputs/afro-xlmr_afr_analysis.json
```

---

## 5. Zulu (ZUL) Experiments

### XLM-RoBERTa
**Single Sentence Analysis:**
```bash
python src/explain.py --model xlm-roberta --lang zul --text "Nginjalo kakhulu" --visualize_type head_view
open outputs/xlm-roberta_zul_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model xlm-roberta --lang zul --identify_cues
open outputs/xlm-roberta_zul_analysis.json
```

### mBERT
**Single Sentence Analysis:**
```bash
python src/explain.py --model mbert --lang zul --text "Nginjalo kakhulu" --visualize_type head_view
open outputs/mbert_zul_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model mbert --lang zul --identify_cues
open outputs/mbert_zul_analysis.json
```

### AfroXLMR
**Single Sentence Analysis:**
```bash
python src/explain.py --model afro-xlmr --lang zul --text "Nginjalo kakhulu" --visualize_type head_view
open outputs/afro-xlmr_zul_attention.html
```

**Dataset Cultural Cues:**
```bash
python src/explain.py --model afro-xlmr --lang zul --identify_cues
open outputs/afro-xlmr_zul_analysis.json
```

---

## 6. Cross-Lingual Transfer Experiments

*These test the "Multilingual Brain" by feeding the exact same text into multiple language models to see how they differ.*

**Test 1: English Transfer (Across XLM-RoBERTa)**
```bash
python src/explain.py --model xlm-roberta --compare_langs amh swa xho afr zul --text "My heart is completely broken"
```

**Test 2: English Transfer (Across mBERT)**
```bash
python src/explain.py --model mbert --compare_langs amh swa xho afr zul --text "My heart is completely broken"
```

**Test 3: English Transfer (Across AfroXLMR)**
```bash
python src/explain.py --model afro-xlmr --compare_langs amh swa xho afr zul --text "My heart is completely broken"
```

**Test 4: African Language Leak (Amharic text fed into all models)**
```bash
python src/explain.py --model xlm-roberta --compare_langs amh swa xho afr zul --text "በጣም አዝኛለሁ"
```
