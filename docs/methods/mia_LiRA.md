---
editor_options: 
  markdown: 
    wrap: 72
---

# Ultimate Hybrid MIA: Multi-Signal Ensemble Attack

## Overview

This method combines four orthogonal signal families into a single
meta-classifier, building on and supersetting the M⁴I-CLIP approach.
While M⁴I-CLIP relies primarily on CLIP cosine similarity (a single
signal family), this attack stacks generation-based metrics, CLIP
embeddings, **LiRA cross-model comparison** (finetuned vs. base
SmolVLM), and generation consistency — capturing membership signals from
both **data characteristics** and **model behavior**.

### Reference Papers

> Hu, T., Wang, R., Jia, J., & Li, B. (2022). *M⁴I: Multi-modal Models
> Membership Inference*. NeurIPS 2022.
>
> Li, Z., Wu, Y., Chen, Y., et al. (2024). *Membership Inference Attacks
> against Large Vision-Language Models*. NeurIPS 2024.
>
> Hu, Y., Li, Z., Liu, Z., et al. (2025). *Membership Inference Attacks
> Against Vision-Language Models*. USENIX Security 2025.
>
> Carlini, N., et al. (2022). *Membership Inference Attacks From First
> Principles*. IEEE S&P 2022.

------------------------------------------------------------------------

## Methodology

### Signal A: Modernized Metric-Based M⁴I (MB-M⁴I)

**Idea:** Generate text from the finetuned SmolVLM for each image
(greedy decoding), then compare to ground-truth text. Members produce
more similar outputs because the model memorized them.

**Features extracted:** - **ROUGE-L, ROUGE-1, ROUGE-2** — N-gram overlap
at different granularities - **Token overlap** — Bag-of-words Jaccard
similarity - **BLEU-1, BLEU-2** — Precision-oriented n-gram overlap
(complementary to ROUGE's recall orientation)

**Difference from M⁴I-CLIP:** We add ROUGE-1, ROUGE-2, and BLEU scores
for richer textual similarity signals beyond just ROUGE-L and token
overlap.

### Signal B: Feature-Based M⁴I with CLIP (FB-M⁴I)

**Idea:** Use CLIP (ViT-B/32) as a cross-modal feature extractor.
Members have tighter image-text alignment in CLIP embedding space
because the VLM overfitted to those specific pairings.

**Features extracted:** - `clip_img_text_sim` — Image ↔ ground-truth
text cosine similarity - `clip_img_gen_sim` — Image ↔ generated text
cosine similarity - `clip_text_gen_sim` — Ground-truth ↔ generated text
cosine similarity - `clip_sim_gap` — Gap between generated and
ground-truth alignment

**Same as M⁴I-CLIP:** This signal family is identical to the colleague's
implementation.

### Signal C: LiRA — Likelihood Ratio Attack 

**Idea:** Compare the finetuned model's loss against the **base
(pre-finetuning) SmolVLM's** loss on each sample. If a sample was in the
finetuning set, the finetuned model's loss dropped more than for
non-members. This cross-model comparison captures a fundamentally
different signal — **how much did finetuning change this sample's
loss?** — that neither CLIP similarity nor generation metrics can
detect.

Based on the LiRA framework from Carlini et al. (2022), adapted for
VLMs.

**Features extracted per model (finetuned + base):** - `loss` — Mean
per-token cross-entropy loss - `perplexity` — exp(loss) - `min_k_prob` —
Average log-probability of the bottom 20% hardest tokens (Shi et al.,
2024) - `max_token_loss` — Worst single token's loss - `std_token_loss`
— Standard deviation across token losses - `zlib_ratio` — Loss
normalized by text compressibility

**Cross-model LiRA features:** - `loss_ratio` — base_loss /
finetuned_loss (higher → more likely member) - `loss_diff` — base_loss −
finetuned_loss - `min_k_diff` — Improvement in Min-K% Prob from base to
finetuned - `perplexity_ratio` — base_perplexity /
finetuned_perplexity - `max_loss_diff` — Improvement in worst-token
loss - `std_loss_diff` — Change in loss variance - `zlib_ratio_diff` —
Change in complexity-normalized loss - `norm_loss_improvement` —
Relative loss drop: (base − ft) / base

**Key difference from M⁴I-CLIP:** It requires loading both the finetuned
and base model simultaneously, but adds an orthogonal membership signal.

### Signal D: Generation Consistency

**Idea:** Generate multiple times with sampling (temperature=1.0) and
measure pairwise similarity across outputs. Members produce **more
consistent** descriptions because the model memorized a specific
response, while non-members produce more varied outputs. Inspired by the
set-level consistency insight from Hu et al. (USENIX Security 2025).

**Features extracted:** - `gen_consistency_mean` — Average pairwise
ROUGE-L across sampled generations - `gen_consistency_std` — Variance in
pairwise similarity

**Key difference from M⁴I-CLIP:** This signal generate multiple times
instead of just once.

### Meta-Classifier

All features (up to 31) are standardized with `StandardScaler` and fed
into an **XGBoost** classifier with tuned hyperparameters (500
estimators, max_depth=4, learning_rate=0.05, colsample_bytree=0.7).

------------------------------------------------------------------------

## Feature Set

### Complete Feature Table (31 features)

| \# | Feature | Family | Description |
|---------------|-----------------|---------------|-------------------------|
| 1 | `rouge_l` | A: Metric | ROUGE-L F1 between generated and ground-truth text |
| 2 | `rouge_1` | A: Metric | ROUGE-1 F1 (unigram overlap) |
| 3 | `rouge_2` | A: Metric | ROUGE-2 F1 (bigram overlap) |
| 4 | `token_overlap` | A: Metric | Jaccard token overlap ratio |
| 5 | `bleu_1` | A: Metric | BLEU-1 (unigram precision) |
| 6 | `bleu_2` | A: Metric | BLEU-2 (bigram precision) |
| 7 | `gen_length` | A: Auxiliary | Word count of generated text |
| 8 | `clip_img_text_sim` | B: CLIP | Cosine similarity: image ↔ ground-truth text |
| 9 | `clip_img_gen_sim` | B: CLIP | Cosine similarity: image ↔ generated text |
| 10 | `clip_text_gen_sim` | B: CLIP | Cosine similarity: ground-truth ↔ generated text |
| 11 | `clip_sim_gap` | B: CLIP | Gap: img↔gen similarity − img↔text similarity |
| 12 | `loss_ft` | C: LiRA | Finetuned model mean loss |
| 13 | `perplexity_ft` | C: LiRA | Finetuned model perplexity |
| 14 | `min_k_prob_ft` | C: LiRA | Finetuned Min-K% Prob (bottom 20% tokens) |
| 15 | `max_token_loss_ft` | C: LiRA | Finetuned worst-token loss |
| 16 | `std_token_loss_ft` | C: LiRA | Finetuned token loss std dev |
| 17 | `zlib_ratio_ft` | C: LiRA | Finetuned loss / zlib compressed length |
| 18 | `loss_base` | C: LiRA | Base model mean loss |
| 19 | `perplexity_base` | C: LiRA | Base model perplexity |
| 20 | `min_k_prob_base` | C: LiRA | Base Min-K% Prob |
| 21 | `max_token_loss_base` | C: LiRA | Base worst-token loss |
| 22 | `std_token_loss_base` | C: LiRA | Base token loss std dev |
| 23 | `zlib_ratio_base` | C: LiRA | Base loss / zlib compressed length |
| 24 | `loss_ratio` | C: LiRA | base_loss / ft_loss |
| 25 | `loss_diff` | C: LiRA | base_loss − ft_loss |
| 26 | `min_k_diff` | C: LiRA | ft_min_k − base_min_k |
| 27 | `perplexity_ratio` | C: LiRA | base_perplexity / ft_perplexity |
| 28 | `max_loss_diff` | C: LiRA | base_max_loss − ft_max_loss |
| 29 | `std_loss_diff` | C: LiRA | base_std − ft_std |
| 30 | `zlib_ratio_diff` | C: LiRA | base_zlib_ratio − ft_zlib_ratio |
| 31 | `norm_loss_improvement` | C: LiRA | (base_loss − ft_loss) / base_loss |
| 32 | `gen_consistency_mean` | D: Consistency | Mean pairwise ROUGE-L across sampled generations |
| 33 | `gen_consistency_std` | D: Consistency | Std dev of pairwise ROUGE-L |

------------------------------------------------------------------------

## Architecture Comparison

| Aspect | M⁴I-CLIP (Colleague) | Ultimate Hybrid (Ours) |
|----------------|---------------------------|-----------------------------|
| Signal families | 2 (Metric + CLIP) | **4** (Metric + CLIP + LiRA + Consistency) |
| Total features | 7 | **up to 33** |
| Models loaded | 1 (finetuned SmolVLM) + CLIP | **2 (finetuned + base SmolVLM)** + CLIP |
| Generation passes | 1 (greedy) | 1 greedy + 2 sampled (configurable) |
| Forward passes | 0 (generation only) | **2 per sample** (finetuned + base logits) |
| Cross-model comparison | None | **LiRA** (loss ratio, loss diff, etc.) |
| Consistency signal | None | **Pairwise ROUGE across sampled outputs** |
| Extra text metrics | ROUGE-L, token overlap | ROUGE-L/1/2, BLEU-1/2, token overlap |
| Classifier | XGBoost | XGBoost (tuned: 500 trees, colsample=0.7) |

------------------------------------------------------------------------

## Results

### Kaggle Leaderboard (AUC)

| Method        | Kaggle AUC  | Notes                      |
|---------------|-------------|----------------------------|
| Baseline      | 0.50082     |                            |
| M⁴I-CLIP LiRA | **0.80548** | 33 features, 18 hrs on MPS |

### Validation Set Metrics

| Method        | Val AUC        | Val [TPR\@FPR](mailto:TPR@FPR){.email}=0.1 |
|---------------|----------------|--------------------------------------------|
| M⁴I-CLIP LiRA | 0.8421 (debug) | 0.4167 (debug)                             |
| M⁴I-CLIP LiRA | 0.8061         | 0.2633                                     |

## Signal Family Ablation

The script includes a built-in ablation study (Cell 10 in the Colab
notebook) that reports AUC for each signal family in isolation and
combined, using Logistic Regression for fair comparison:

| Family | Features | Expected Signal |
|------------------|--------------------|----------------------------------|
| A (Metric only) | ROUGE, BLEU, overlap | Generation similarity to GT |
| B (CLIP only) | CLIP cosine similarities | Data-level image-text alignment |
| A+B (≈ Colleague) | Metric + CLIP | Baseline to beat |
| C (LiRA cross) | Loss ratio, diff, etc. | Model-level finetuning signal |
| D (Consistency) | Pairwise generation ROUGE | Output stability |
| ALL (Ultimate) | All 33 features | Full ensemble |

------------------------------------------------------------------------

### Speed Optimization

``` python
CONSISTENCY_REPS = 0    # Skip consistency generations (biggest speedup)
MAX_GEN_TOKENS   = 150  # Shorter generations
```

### Output Files

-   `output/ultimate_submission.csv` — Kaggle submission
-   `output/ultimate_features.csv` — Detailed test features
-   `ultimate_cache/*.json` — Cached features per split (resumable)

------------------------------------------------------------------------

## Dependencies

```         
pip install rouge-score open_clip_torch xgboost nltk scikit-learn datasets transformers accelerate Pillow
```
