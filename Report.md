% ============================================================
% Group 51 – COS 760 – University of Pretoria – 2026
% Submission filename: Group51_u23671964.pdf
%
% To compile in Overleaf:
%   1. Create a new project from the ACL 2023 template (provides acl.sty).
%   2. Upload this file (rename to Group51_u23671964.tex) and custom.bib.
%   3. Place the three bar-chart images in the same directory as the .tex file:
%        chart1_model.png    (Model performance: Macro-F1 & Jaccard per model)
%        chart2_language.png (Language performance: Macro-F1 & Jaccard per language)
%        chart3_strategy.png (Strategy comparison: Macro-F1 & Jaccard per training strategy)
%   4. Set the compiler to pdfLaTeX and build.
% ============================================================

\documentclass[11pt]{article}

\usepackage{acl}

\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}

\title{Multilabel Emotion Analysis for African Languages:\\
Transfer Learning and Data Augmentation on BRIGHTER \& EthioEmo}

\author{
  \textbf{Nicholas Dobson\textsuperscript{1}},
  \textbf{Simon van der Merwe\textsuperscript{1}},
  \textbf{Naazneen Khan\textsuperscript{1}}
\\
  \textsuperscript{1}University of Pretoria, Department of Computer Science
\\
  \small{Group 51 COS 760 \quad
  u23671964 \quad u04576617 \quad u22527533}
}

\begin{document}
\maketitle

% 
\begin{abstract}
% 
Multilabel emotion classification remains an open challenge for African languages,
which are severely underrepresented in NLP research.
This paper presents experiments on three African languages, Afrikaans (\texttt{afr}),
Amharic (\texttt{amh}), and Swahili (\texttt{swa})using the BRIGHTER and EthioEmo
datasets. We fine-tune three multilingual pre-trained models: mBERT, XLM-RoBERTa-base, and AfroXLMR under 3-fold cross-validation with per-class threshold optimisation.
We investigate two data augmentation strategies, back-translation via NLLB-200 and
paraphrasing via the Gemini API, as well as their combination.
AfroXLMR consistently achieves the best macro-F1 across all languages
(0.400 for Afrikaans, 0.668 for Amharic, 0.316 for Swahili), confirming the value of
Africa-focused pretraining. Counter to expectations, neither augmentation strategy
reliably improves upon the no-augmentation baseline.
Per-class analysis reveals \textit{surprise} as the hardest emotion for Afrikaans
(F1\,=\,0.00) and \textit{fear} as the hardest for Swahili (F1\,=\,0.10).
Attention-weight visualisation confirms that the model relies on emotion-bearing
lexical cues and highlights joy--sadness co-prediction as a key failure mode.
All fine-tuned models substantially outperform the majority-label baseline.
\end{abstract}

% 
\section{Introduction}
% 
Africa is home to over 2{,}000 languages spanning six distinct language families,
yet the vast majority of NLP research focuses on a small set of high-resource
languages~\citep{muhammad2025brighter}. Of the 75 African languages with at least
one million speakers, very few have dedicated emotion analysis systems, limiting the
reach of affective computing in healthcare, social media moderation, education, and
mental health. Emotion classification, assigning one or more emotion labels to a
text, is particularly underdeveloped for African languages, despite clear real-world
relevance and the growing presence of African language content on social media.

This project addresses multilabel emotion classification for African languages by
leveraging two publicly available, human-annotated datasets:
BRIGHTER~\citep{muhammad2025brighter}, covering 28 languages across seven language families, and EthioEmo~\citep{belay2025a}, covering four Ethiopian languages.
Together, they provide binary annotations over six emotion categories:
\textit{anger}, \textit{fear}, \textit{surprise}, \textit{sadness}, \textit{joy},
and \textit{disgust}. Each instance is a short text (typically a social media post
or news sentence) with independent binary labels per emotion.

Our contributions are threefold: (1) we establish fine-tuned baselines for
Afrikaans, Amharic, and Swahili using three multilingual transformer models under
rigorous 3-fold cross-validation; (2) we systematically evaluate two data
augmentation strategies NLLB-200, back-translation, and Gemini API paraphrasing and
their combination; and (3) we conduct per-class error analysis with attention-weight
visualisation to illuminate model behaviour across linguistically diverse languages.

We investigate three research questions aligned with our project proposal:
\begin{itemize}
  \item \textbf{RQ1:} Which multilingual pre-trained model, mBERT, XLM-RoBERTa,
        or AfroXLMRbest supports multilabel emotion classification across
        Afrikaans, Amharic, and Swahili?
  \item \textbf{RQ2:} How do data augmentation strategies (NLLB-200
        back-translation and Gemini paraphrasing) affect macro-F1 in
        low-resource settings?
  \item \textbf{RQ3:} What do per-class results and attention visualisations
        reveal about linguistically and culturally specific prediction challenges?
\end{itemize}

All fine-tuned models substantially exceed the majority-label baseline, confirming
that cross-lingual transfer provides a strong signal even in low-resource settings.
However, augmentation did not yield the anticipated 2--5\% gains, raising important
questions about noise injection and threshold sensitivity in small-data regimes.

% 
\section{Related Work}
% 
Emotion analysis has historically concentrated on English and European
languages~\citep{paran2025}. The BRIGHTER dataset~\citep{muhammad2025brighter}
and EthioEmo~\citep{belay2025a} mark a significant expansion toward African and
low-resource language coverage, providing multilabel annotations consistent with
the SemEval-2025 Task\,11 shared task format~\citep{paran2025}.
Participating systems employed fine-tuned transformers, cross-lingual transfer,
and zero-shot prompting, establishing competitive baselines against which our fine-tuned results can be contextualised.

For African NLP broadly, the AfriSenti benchmark~\citep{muhammad2023} provided
Twitter-based sentiment data for 14 African languages, and demonstrated that
domain-adapted models substantially outperform zero-shot and few-shot LLMs.
AfroXLMR~\citep{alabi2022} advances this paradigm by applying multilingual adaptive
fine-tuning of XLM-RoBERTa~\citep{conneau2020} on a large African language corpus,
yielding improvements across named entity recognition, text classification, and
sentiment. Language-targeted fine-tuning is critical for lower-resource settings, with models trained on focused single-language data consistently outperforming those trained on pooled multilingual corpora~\citep{belay2025a}.

Data augmentation for low-resource NLP has produced mixed results in prior work.
Back-translation using neural MT systems such as NLLB-200~\citep{costa2022} can
expand training set sizes but introduces translation errors that are particularly
pronounced for morphologically complex languages with limited parallel data.
Paraphrasing via large language models offers a complementary augmentation approach, though hallucination risks and semantic drift remain practical concerns. Filtering
augmented samples by cosine similarity with a multilingual sentence embeddings~\citep{reimers2019} is a standard mitigation strategy, though the
quality of the resulting samples is sensitive to the similarity threshold and the
alignment between source and target embedding spaces. To our knowledge, no prior work has applied systematic augmentation evaluation to African emotion classification datasets, a gap our study directly addresses.

% 
\section{Datasets}
% 
\paragraph{BRIGHTER.}
We use BRIGHTER~\citep{muhammad2025brighter}, comprising approximately 100{,}000
short texts in 28 languages with binary multilabel annotations for six emotion
categories. We use the Afrikaans (\texttt{afr}) and Swahili (\texttt{swa}) subsets,
loaded via HuggingFace Datasets with official train/dev/test splits. BRIGHTER uses
a ``dev'' split name rather than ``validation'', which our pipeline normalises at
load time.

\paragraph{EthioEmo.}
EthioEmo~\citep{belay2025a} provides 23{,}441 instances across four Ethiopian
languages annotated for the same six emotion categories. We use the Amharic
(\texttt{amh}) subset with its 60/30/10 train/dev/test split, making Amharic the
largest single-language training set in our experiments.

\paragraph{Preprocessing.}
All texts are Unicode-normalised (NFC) and deduplicated within each split.
Class imbalance is handled via per-class positive-weight scaling in the loss:
$w_j = n_{\mathrm{neg},j} / n_{\mathrm{pos},j}$, ensuring rare emotions such
as \textit{surprise} and \textit{disgust} are not overwhelmed by frequent ones.
Approximate split sizes before cross-validation merging are shown in
Table~\ref{tab:datastats}. The emotion label distribution is skewed across all
three languages: \textit{joy} and \textit{sadness} are the most prevalent, while
\textit{surprise} is nearly absent in Afrikaans.

\begin{table}[t]
  \centering
  \small
  \begin{tabular}{llccc}
    \toprule
    \textbf{Lang.} & \textbf{Source} & \textbf{Train} & \textbf{Val} & \textbf{Test} \\
    \midrule
    \texttt{afr} & BRIGHTER  & $\sim$490  & $\sim$98   & $\sim$196 \\
    \texttt{amh} & EthioEmo  & $\sim$3{,}500 & $\sim$1{,}750 & $\sim$580 \\
    \texttt{swa} & BRIGHTER  & $\sim$1{,}500 & $\sim$300  & $\sim$300 \\
    \bottomrule
  \end{tabular}
  \caption{Approximate dataset split sizes per language before cross-validation
  merging and before augmentation. Afrikaans is the most data-scarce setting.}
  \label{tab:datastats}
\end{table}

% 
\section{Methodology}
% 
\subsection{Model Architecture and Training}

We fine-tune three multilingual pre-trained models:
\textbf{mBERT} (\texttt{bert-base-multilingual-cased};~\citealt{devlin2019}),
\textbf{XLM-RoBERTa} (\texttt{xlm-roberta-base};~\citealt{conneau2020}), and
\textbf{AfroXLMR} (\texttt{Davlan/afro-xlmr-base};~\citealt{alabi2022}).
Each model is extended with a single linear classifier over the \texttt{[CLS]}
representation, producing six independent sigmoid-activated outputs for multilabel
prediction:
\begin{equation}
  \hat{y}_j = \sigma\!\bigl(W_j\,\mathbf{h}_{\texttt{[CLS]}}\bigr),\quad
  j \in \{1,\ldots,6\}
\end{equation}

Training uses \texttt{BCEWithLogitsLoss} with per-class positive weighting, AdamW
($\mathrm{lr} = 2\!\times\!10^{-5}$, batch size 16, max 5 epochs), and a cosine
learning-rate schedule with 10\% linear warm-up. Gradient clipping
($\|\nabla\|_2 \le 1.0$) is applied at every update step to stabilise training
on small datasets. Early stopping is triggered after two consecutive epochs without
improvement in validation macro-F1, and the best-epoch checkpoint is retained.

\subsection{Per-Class Threshold Optimisation}

Rather than applying a fixed 0.5 decision threshold to all classes, we optimise
per-class thresholds on the validation fold after training is complete. For each
emotion $j$, we sweep candidate thresholds $\tau \in [0.2, 0.8]$ at steps of 0.025
and select the value maximising per-class F1:
\begin{equation}
  \tau_j^\star = \operatorname*{arg\,max}_{\tau \in [0.2, 0.8]}\;
  F_1\!\bigl(y_j,\;\mathbb{1}[\hat{y}_j \ge \tau]\bigr)
\end{equation}
This step addresses a key failure mode: for high-frequency emotions such as
\textit{joy}, the model often predicts both \textit{joy} and \textit{sadness} above
0.5 for ambiguous texts. Raising the threshold for over-predicted classes suppresses
false positives without requiring architectural changes. Per-class thresholds are
saved and reused during test evaluation and attention visualisation.

\subsection{Data Augmentation}

\paragraph{Stage\,2a -- NLLB Back-Translation.}
We use \texttt{facebook/nllb-200-distilled-600M}~\citep{costa2022} to translate
each training sentence from the source language to English and then back. Augmented samples are retained only if their cosine similarity to the original is computed with \texttt{paraphrase-multilingual-MiniLM-L12-v2}~\citep{reimers2019}
exceeds 0.75, and if the sample is not identical to the source. This threshold was
lowered from the proposed 0.85 to accommodate the cross-lingual embedding alignment gap between African languages and their English translations.

\paragraph{Stage\,2b -- Gemini Paraphrasing.}
We call the Gemini API (\texttt{gemini-1.5-flash}) in batches of 20, prompting the
model to paraphrase each sentence in its native language while strictly preserving
emotional meaning and tone. A JSON-schema output constraint enforces clean structured responses, and the same cosine-similarity filter ($\ge$0.75) is applied.
This approach replaces the multilingual T5 paraphrasing proposed initially,
offering stronger instruction-following for morphologically diverse African languages.
API calls are rate-limited to 15 requests per minute.

\paragraph{Stage\,2c -- Combined.}
The back-translated and Gemini-paraphrased augmented samples are concatenated
(excluding the shared baseline) to form a combined training set, drawing on both
augmentation sources.

\subsection{Evaluation Protocol}

To address small validation sets, particularly Afrikaans (${\approx}98$ validation
examples), we apply \textbf{3-fold cross-validation} on the merged train+dev pool
per language, evaluating on the fixed held-out test set. This yields three independent training runs per condition, providing the mean and standard deviation of macro-F1 and macro Jaccard similarity as primary metrics, consistent with SemEval-2025 Task\,11~\citep{paran2025}.

% 
\section{Results}
% 
Table~\ref{tab:results} reports macro-F1 (mean\,$\pm$\,std, 3-fold CV) for all
conditions. Figures~\ref{fig:model_lang} and~\ref{fig:strategy} visualise aggregate
model, language, and strategy comparisons across all folds. Per-class F1 for the
best-performing configuration (AfroXLMR, baseline training) is shown in
Table~\ref{tab:perclass}.

The majority-label baseline predicts \textit{joy} as the most frequent emotion for
every instance in Afrikaans (macro-F1\,=\,0.090) and Amharic (0.045), and all-zero
labels in Swahili (0.000), where no single emotion dominates the label distribution.
All fine-tuned models substantially exceed this ceiling across all languages,
validating the utility of transfer learning even in the most data-scarce setting
(Afrikaans, ${\approx}490$ training examples).

\begin{table*}[t]
  \centering
  \small
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{llccc}
    \toprule
    \textbf{Training Strategy} & \textbf{Model} &
    \textbf{Afrikaans (afr)} & \textbf{Amharic (amh)} & \textbf{Swahili (swa)} \\
    \midrule
    Majority Baseline &  & 0.090 & 0.045 & 0.000 \\
    \midrule
    \multirow{3}{*}{Baseline (no augmentation)}
      & mBERT    & $0.326 \pm 0.026$ & $0.306 \pm 0.001$ & $0.251 \pm 0.002$ \\
      & XLM-R    & $0.374 \pm 0.017$ & $0.647 \pm 0.017$ & $0.281 \pm 0.001$ \\
      & AfroXLMR & $\mathbf{0.400} \pm 0.008$ & $\mathbf{0.668} \pm 0.006$ & $\mathbf{0.316} \pm 0.008$ \\
    \midrule
    \multirow{3}{*}{NLLB Back-Translation}
      & mBERT    & $0.292 \pm 0.010$ & $0.300 \pm 0.006$ & $0.238 \pm 0.010$ \\
      & XLM-R    & $0.354 \pm 0.010$ & $0.641 \pm 0.004$ & $0.277 \pm 0.008$ \\
      & AfroXLMR & $0.383 \pm 0.009$ & $0.664 \pm 0.005$ & $0.294 \pm 0.023$ \\
    \midrule
    \multirow{3}{*}{Gemini Paraphrasing}
      & mBERT    & $0.292 \pm 0.017$ & $0.305 \pm 0.001$ & $0.233 \pm 0.013$ \\
      & XLM-R    & $0.342 \pm 0.012$ & $0.644 \pm 0.014$ & $0.273 \pm 0.008$ \\
      & AfroXLMR & $0.384 \pm 0.011$ & $0.667 \pm 0.003$ & $0.304 \pm 0.001$ \\
    \midrule
    \multirow{3}{*}{Combined (BT + Gemini)}
      & mBERT    & $0.303 \pm 0.008$ & $0.300 \pm 0.006$ & $0.220 \pm 0.009$ \\
      & XLM-R    & $0.361 \pm 0.017$ & $0.652 \pm 0.001$ & $0.264 \pm 0.009$ \\
      & AfroXLMR & $0.377 \pm 0.013$ & $0.663 \pm 0.001$ & $0.309 \pm 0.021$ \\
    \bottomrule
  \end{tabular}
  \caption{Macro-averaged F1 (mean\,$\pm$\,std, 3-fold CV) for all training conditions.
  \textbf{Bold} marks the best result per language column.
  The majority baseline predicts the most frequent label set seen in training.}
  \label{tab:results}
\end{table*}

\begin{table}[t]
  \centering
  \small
  \begin{tabular}{lccc}
    \toprule
    \textbf{Emotion} & \textbf{afr} & \textbf{amh} & \textbf{swa} \\
    \midrule
    Anger   & 0.371 & 0.662 & 0.292 \\
    Disgust & 0.224 & 0.741 & 0.212 \\
    Fear    & 0.455 & 0.497 & 0.098 \\
    Joy     & 0.707 & 0.742 & 0.523 \\
    Sadness & 0.643 & 0.710 & 0.332 \\
    Surprise& 0.000 & 0.657 & 0.441 \\
    \midrule
    \textit{Macro} & \textit{0.400} & \textit{0.668} & \textit{0.316} \\
    \bottomrule
  \end{tabular}
  \caption{Per-class F1 (mean, 3-fold CV) for AfroXLMR under baseline training.
  \textit{Surprise} in Afrikaans receives no useful signal; \textit{fear} in
  Swahili is the most challenging category.}
  \label{tab:perclass}
\end{table}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.48\linewidth]{chart1_model_performance_clean.png}
  \hfill
  \includegraphics[width=0.48\linewidth]{chart2_language_performance_clean.png}
  \caption{
    \textbf{Left:} Average macro-F1 and Jaccard by model, across all languages and
    training strategies. AfroXLMR leads on both metrics.
    \textbf{Right:} Average macro-F1 and Jaccard by language, across all models and
    training strategies. Amharic achieves the highest scores, benefiting from
    EthioEmo's focused, larger training set.}
  \label{fig:model_lang}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{chart3_training_strategy_clean.png}
  \caption{Macro-F1 and Jaccard by training strategy, averaged across all models
  and languages. The baseline (no augmentation) matches or outperforms all
  augmented variants on both metrics.}
  \label{fig:strategy}
\end{figure}

% 
\section{Discussion}
% 

\subsection{RQ1: Model Comparison}

AfroXLMR achieves the highest macro-F1 across all three languages and all training
conditions (Figure~\ref{fig:model_lang}, left), confirming our proposal hypothesis.
The advantage is most pronounced for Amharic: AfroXLMR scores 0.668 versus
mBERT's 0.306a 36 percentage-point gap. mBERT's multilingual vocabulary assigns
relatively few subword tokens to Amharic's Ge'ez script, creating an embedding
bottleneck that AfroXLMR's Africa-focused continued pretraining~\citep{alabi2022}
resolves through exposure to the Ethiopian language corpora. XLM-RoBERTa performs
competitively on Amharic (0.647) but falls consistently below AfroXLMR across all
three languages, despite a substantially larger general pretraining corpus, suggesting
that domain specificity outweighs raw data scale for low-resource African languages.

Afrikaans shows the smallest inter-model gap (AfroXLMR 0.400, mBERT 0.326), attributable to its tiny training set (${\approx}490$ examples) limiting fine-tuning effectiveness equally across models. Swahili occupies an intermediate position, benefiting from a larger training set than Afrikaans while underperforming Amharic due to EthioEmo's more homogeneous annotation scope.

\subsection{RQ2: Augmentation Strategies}

Counter to the 2--5\% macro-F1 improvement hypothesised in our proposal, all three
augmentation strategies produce marginal decreases relative to the no-augmentation
baseline across most conditions (Figure~\ref{fig:strategy}). For AfroXLMR on
Amharic, the best augmented result (Gemini paraphrasing: 0.667), matches the
baseline (0.668) within rounding error. Augmentation is most detrimental for
mBERT on Swahili (baseline 0.251 vs.\ Combined 0.220), suggesting weaker models
are more sensitive to label noise. Three factors explain these results:

\begin{enumerate}
  \item \textbf{Threshold calibration noise.} Per-class threshold optimisation is
        performed on small validation sets (${\approx}98$ Afrikaans examples).
        Augmenting the training set without a commensurate increase in validation
        size shifts the model's output distribution while providing an insufficient
        calibration signal for threshold selection.
  \item \textbf{Semantic drift in augmented samples.} Lowering the cosine-similarity
        filtering from the proposed 0.85 to 0.75 to accommodate cross-lingual embedding
        spaces may admit paraphrases where subtle meaning shifts alter the correct
        emotion label, particularly for culturally loaded or context-dependent
        expressions. NLLB translation quality also degrades for low-resource African
        languages with limited parallel training data~\citep{costa2022}.
  \item \textbf{Compounded noise in the Combined strategy.} Concatenating BT and
        Gemini augmentations do not outperform either source alone on any
        language-model pair. This indicates that diverse noise sources compound
        without introducing complementary variation under the current filtering regime.
\end{enumerate}

\subsection{RQ3: Per-Class Analysis and Attention Visualisation}

\paragraph{Per-class patterns.}
Table~\ref{tab:perclass} reveals striking variation across emotions and languages.
For Afrikaans, \textit{surprise} achieves F1\,=\,0.00 across all models and
conditions, a complete prediction failure attributable to near-total annotation
sparsity in the BRIGHTER Afrikaans subset. The majority classifier also produces
zero \textit{surprise} predictions, confirming that no model can recover a useful
signal from the handful of positive examples. In contrast, \textit{joy}
(F1\,=\,0.707) and \textit{sadness} (F1\,=\,0.643) are the most learnable
categories, consistent with their higher prevalence and the availability of
distinct lexical cues in Afrikaans social media text.

For Swahili, \textit{fear} is the hardest class (F1\,=\,0.098) despite reasonable
performance on other emotions (\textit{joy}: 0.523, \textit{surprise}: 0.441).
This pattern likely reflects culturally specific expressions of fear in Swahili
that rely on implicit contextual cues or idioms not well-captured by cross-lingual
embeddings trained primarily on higher-resource language data. Amharic achieves the
most balanced per-class profile (all F1\,$\ge$\,0.497), driven by EthioEmo's
larger, language-focused training set, consistent with the finding that per-language data volume is the primary driver of per-class classification coverage.

\paragraph{Attention visualisation.}
Analysis of AfroXLMR \texttt{[CLS]} attention patterns confirms that the model attends primarily to emotion-bearing content words (evaluative adjectives, emotion-denoting verbs) rather than function words, consistent across Afrikaans and Amharic test examples.
A key failure mode is the co-prediction of \textit{joy} and \textit{sadness}: texts expressing mixed affect sometimes activate both outputs above 0.5 simultaneously. Per-class threshold optimisation partially resolves this, though calibration is constrained by Afrikaans's small validation set, confirming that threshold post-processing is a necessary complement to fine-tuning in low-resource settings.

% 
\section{Conclusion}
% 
We presented a systematic study of multilabel emotion classification for three
African languages, Afrikaans, Amharic, and Swahili, using three multilingual
transformer architectures and four training conditions. AfroXLMR consistently
outperforms mBERT and XLM-RoBERTa under all conditions, with the largest gains
for Amharic, confirming that Africa-focused pretraining substantially outweighs
raw model scale for low-resource African language tasks. Data augmentation via
NLLB back-translation and Gemini paraphrasing did not reliably improve macro-F1,
primarily due to threshold calibration noise from small validation sets, semantic
drift in augmented samples, and compounded noise when combining sources.

Per-class analysis identifies \textit{surprise} in Afrikaans (F1\,=\,0.00) and
\textit{fear} in Swahili (F1\,=\,0.10) as the hardest classes, reflecting data
sparsity and culturally specific expression patterns, respectively. Attention
visualisation confirms that models rely on emotion-bearing lexical cues and
highlights joy--sadness co-prediction as a key failure mode that threshold
optimisation partially resolves. These findings provide a reproducible baseline
and practical pipeline for future African language emotion classification work.

Future work should explore language-specific augmentation with human quality evaluation and extension to additional African languages not covered by either dataset.

% 
\section*{Limitations}
% 
Our experiments cover three of the 17 African languages available across the two
datasets, due to computational constraints. The Afrikaans training set is very
small (${\approx}490$ examples), making per-class threshold optimisation
statistically unreliable and fold-level results sensitive to random seed variation
(std up to 0.026). NLLB-200 back-translation quality is not human-evaluated,
and cosine-similarity filtering does not guarantee label preservation for
morphologically complex languages. Results are not directly comparable to
SemEval-2025 Task\,11 leaderboard entries, which use different official splits,
and attention-weight analysis provides qualitative insights only.

% 
\bibliography{custom}
% 

\end{document}
