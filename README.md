#Multilabel Emotion Analysis for African Languages Using Transfer Learning and Data Augmentation on BRIGHTER + EthioEmo
##Nicholas Dobson, Simon van der Merwe, Naazneen Khan |  Group 51
###COS 760 – University of Pretoria 2026
##1. Introduction
Natural Language Processing (NLP) research has achieved remarkable progress over the past decade, yet the vast majority of advances have been concentrated in a small set of high-resource languages. Africa, despite being home to over 2,000 languages from more than six distinct language families representing the world’s highest linguistic diversity, remains underrepresented in NLP research (Muhammad et al., 2025). Of the 75 African languages with at least one million speakers, very few have dedicated emotion analysis systems, limiting the reach of affective computing applications in healthcare, social media moderation, education, and mental health.
This project addresses the problem of multilabel emotion classification for African languages using the BRIGHTER + EthioEmo datasets, which collectively provide human-annotated, multilabel emotion data for 32 languages, including 17 African languages (Muhammad et al., 2025; Belay et al., 2025a). We investigate two key strategies for improving model performance in low-resource settings: (1) transfer learning via fine-tuning multilingual pre-trained models such as AfroXLMR and XLM-RoBERTa, and (2) data augmentation through back-translation and paraphrasing.
Our contributions are threefold: we establish fine-tuned baselines for a focused subset of African languages, we systematically evaluate data augmentation strategies in low-resource multilabel settings, and we conduct an explanatory error analysis using attention visualisation to illuminate model behaviour across linguistically diverse languages. The societal motivation is clear: equitable NLP tools for African languages support digital inclusion, enable culturally sensitive AI, and address longstanding fairness gaps in the field.
##2. Background and Related Work
Emotion analysis is a well-established NLP task, but most benchmarks, such as SemEval emotion datasets, have historically focused on English or a handful of European languages. The introduction of BRIGHTER (Muhammad et al., 2025), a multilabel emotion-annotated dataset spanning 28 languages, marked a significant step toward multilingual emotion recognition. Complementing this, EthioEmo (Belay et al., 2025a) extends coverage to four Ethiopian languages (Amharic, Tigrinya, Oromo, and Somali), evaluating large language model (LLM) capabilities for low-resource multilabel emotion classification.
Prior work on African language NLP has centred primarily on sentiment analysis. The AfriSenti benchmark (Muhammad et al., 2023) provides Twitter-based sentiment data for 14 African languages and demonstrated that fine-tuning AfroXLMR substantially outperforms zero-shot and few-shot LLM approaches for low-resource languages. SemEval-2025 Task 11 (Paran et al., 2025; Poulaei et al., 2025) extended this to multilabel emotion classification, with participating systems exploring fine-tuned transformers and cross-lingual transfer, establishing competitive baselines. Belay et al. (2025b) further explore multi-label emotion intensity prediction for Ethiopian languages, finding that multilingual models benefit from language-targeted fine-tuning.
A critical gap in prior work is the absence of systematic data augmentation studies for African language emotion datasets, where training sets are typically small (often under 1,000 labelled examples per language). Additionally, explanatory evaluation using attention analysis or SHAP values has not been applied in the African NLP emotion context. Our project will target both those gaps directly.
Responsible NLP Reflection
Prior datasets may carry annotator bias, particularly where emotional expression is culturally contextual (e.g., concepts of ‘joy’ or ‘fear’ differ across cultures). BRIGHTER mitigates this by using fluent native-speaker annotators, but inter-annotator agreement is not perfect. Our work acknowledges these limitations and uses error analysis to surface culturally-linked misclassifications.
##3. Proposed Methodology
###3.1 Research Questions
•	RQ1: Which multilingual pre-trained model (mBERT, AfroXLMR, XLM-RoBERTa) best supports multilabel emotion classification across the selected African languages?
•	RQ2: How do data augmentation strategies (back-translation, paraphrasing) affect F1 performance in low-resource language settings?
•	RQ3: What do attention visualisations reveal about the linguistic features models use when predicting emotion labels?
###3.2 Dataset
We use the BRIGHTER + EthioEmo datasets (Muhammad et al., 2025; Belay et al., 2025a), publicly available via HuggingFace and the shared task repository. The combined dataset covers 32 languages with multilabel annotations across six emotion categories: anger, fear, surprise, sadness, happiness, and disgust. Each instance is a short text (typically a social media post or news sentence) with binary labels per emotion. We focus on the following six languages selected for linguistic diversity and data availability:
•	BRIGHTER dataset: with 100,000 emotion-annotated instances in 28 languages from 7 distinct language families
•	EthioEmo dataset: 23,441 instances in four languages
Where available, we supplement with AfriSenti sentiment data as an auxiliary training signal. All datasets are openly licensed for research use. Data will be preprocessed to normalise Unicode, remove duplicates, and handle class imbalance using weighted loss functions.
###3.3 Approach
Our pipeline has three stages:
Stage 1 – Baseline: We fine-tune three multilingual models: mBERT, XLM-RoBERTa-base, and AfroXLMR on each language’s training split with a multilabel sigmoid output layer. We use binary cross-entropy loss with class weighting to handle label imbalance. Hyperparameters: learning rate 2e-5, batch size 16, 5 epochs, early stopping on validation F1.
Stage 2 – Data Augmentation: We apply back-translation (source language → English → source language via Helsinki-NLP MarianMT or the Google Translate API) to double the training set size. We additionally test paraphrasing using a multilingual T5 model. Augmented data is filtered by semantic similarity (cosine similarity > 0.85) to ensure label consistency.
Stage 3 – Explanatory Evaluation: We apply attention-weight visualisation on the best-performing model to identify which tokens drive emotion predictions. We use BERTViz to extract and compare attention patterns across languages, identifying cross-lingual patterns and culturally specific features.
Responsible NLP Reflection
Back-translation may introduce cultural drift or disfluencies for morphologically rich languages like isiZulu. We mitigate this with similarity filtering. All model checkpoints will be released for reproducibility. We follow BRIGHTER’s data-use licence terms.
###4. Evaluation
Success is measured using the following metrics, consistent with SemEval-2025 Task 11 standards:
•	Macro-averaged F1 (primary): accounts for class imbalance across six emotion labels.
•	Precision and Recall per emotion class: to identify per-label model weaknesses.
•	Jaccard Similarity Score: for multilabel set overlap.
•	Pearson r: for emotion intensity sub-task (if explored).
Baselines include: (1) a majority-label classifier, (2) mBERT fine-tuned without augmentation, and (3) results reported in SemEval-2025 Task 11 system papers (Paran et al., 2025; Poulaei et al., 2025) where comparable languages are available. We evaluate separately per language and report aggregate scores to assess cross-lingual consistency. Error analysis will categorise failure modes by emotion label, language, and text domain.
###5. Expected Output and Contributions
We expect AfroXLMR to outperform mBERT and XLM-RoBERTa on African languages, consistent with findings in AfriSenti (Muhammad et al., 2023). Data augmentation is hypothesised to yield 2–5% macro-F1 gains for the lowest-resource languages, with diminishing returns for slightly larger datasets.
Explanatory analysis is expected to reveal that models rely on culturally-specific lexical cues for emotion, and that misclassifications cluster around ambiguous, mixed-emotion texts. Key contributions include: fine-tuned model weights for African languages, a reproducible augmentation pipeline, and an attention-based error analysis catalogue.
Feasibility and Work Plan
•	Weeks 1–2: Data acquisition, preprocessing, baseline experiments
•	Weeks 3–4: Augmentation pipeline implementation and experiments
•	Weeks 5–6: Attention visualisation, error analysis, and write-up
All required datasets are publicly available. The primary risk is translation quality for back-augmentation; this is mitigated by similarity filtering. Team roles are shared, with weekly meetings, and are divided as follows:
•	Nicholas: Baseline Modelling & Evaluation
•	Simon: Data Augmentation Pipeline
•	Naaz: Explanatory Analysis & Write-up
###6. References
Belay, T. D., Azime, I. A., Ayele, A. A., Sidorov, G., Klakow, D., Slusallek, P., & Yimam, S. M. (2025a). Evaluating the capabilities of large language models for multi-label emotion understanding. In Proceedings of the 31st International Conference on Computational Linguistics (pp. 3523–3540).

Belay, T. D., Gete, D. K., Ayele, A. A., Kolesnikova, O., Ameer, I., Sidorov, G., & Yimam, S. M. (2025b). Enhancing multi-label emotion analysis and corresponding intensities for Ethiopian languages. arXiv preprint arXiv:2503.18253.

Muhammad, S. H., Ousidhoum, N., Abdulmumin, I., Wahle, J. P., Ruas, T., Beloucif, M., & Mohammad, S. (2025). BRIGHTER: Bridging the gap in human-annotated textual emotion recognition datasets for 28 languages. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 8895–8916).

Muhammad, S. H., Abdulmumin, I., Ayele, A. A., Ousidhoum, N., Adelani, D. I., Yimam, S. M., & Mohammad, S. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 13968–13981).

Paran, A. I., Aftahee, S., Hossan, M. R., Hossain, J., & Hoque, M. M. (2025). Zero_Shot at SemEval-2025 Task 11: Fine-tuning deep learning and transformer-based models for emotion detection in multi-label classification, intensity estimation, and cross-lingual adaptation. In Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025) (pp. 1890–1904).

Poulaei, M. S., Zare, M. E., Mohammadi, M. R., & Eetemadi, S. (2025). YNWA_PZ at SemEval-2025 Task 11: Multilingual multi-label emotion classification. In Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025) (pp. 508–521).
