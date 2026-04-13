# Neighborhood + MIA Combined Approach for VLM Membership Inference

## Overview

This document describes our combined attack pipeline that merges two complementary membership inference strategies to detect whether an (image, caption) pair was used during fine-tuning of a Vision Language Model (SmolVLM-256M-Instruct).

-   **Label 1 (member):** The model saw this exact (image, caption) pair during fine-tuning.
-   **Label 0 (non-member):** Any other combination.

Our best result combines features from two independent methods: - **Milestone 6 — Three-Layer MIA Attack** (3 features): loss ratio, caption contrast, corruption sensitivity - **Milestone 5 — Dual-Model Neighborhood Attack** (17 features): text/image perturbation-based signals

Final model: **Logistic Regression** with `class_weight='balanced'` on 20 combined features.

------------------------------------------------------------------------

## Method 1: Three-Layer MIA Attack (Milestone 6)

Source: `milestone6/src/mia_attack.py`

This method extracts three features by probing the finetuned model's behavior under different perturbation conditions. All three rely on computing cross-entropy loss on the assistant (caption) tokens only, using SmolVLM's chat template.

### Feature 1: Loss Ratio (`loss_ratio`)

$$\text{loss\_ratio} = \frac{\text{loss}_{\text{base}}}{\text{loss}_{\text{finetuned}} + \epsilon}$$

**Intuition:** The finetuned model memorizes training pairs, producing lower loss than the base model for members. The base model acts as a control — it eliminates confounding from inherently "easy" or "hard" samples. This is a differential signal: only memorized pairs show a significantly lower finetuned loss relative to the base model.

**Why it works:** This is our strongest single feature (AUC = 0.7929). Members have a **lower** loss ratio (mean 1.23 vs 1.57 for non-members), because the finetuned model's loss drops disproportionately for memorized pairs while the base model's loss stays similar.

### Feature 2: Caption Contrast (`caption_contrast`)

$$\text{caption\_contrast} = \frac{1}{N}\sum_{i=1}^{N} \text{loss}(x, c_i^{\text{shuffled}}) - \text{loss}(x, c^{\text{original}})$$

**Intuition:** Fix the image. Replace the original caption with N randomly sampled captions from the dataset. If the model memorized this specific (image, caption) binding, the original caption should have significantly lower loss than random alternatives.

**Why it is weak (AUC = 0.5085):** The VLM generalizes well to any correct image-caption pairing, so even non-member pairs show reasonable loss with their original caption. The gap between original and shuffled captions is not specific enough to memorization.

### Feature 3: Corruption Sensitivity (`corruption_sensitivity`)

$$\text{corruption\_sensitivity} = -(\text{loss}_{\text{corrupted}} - \text{loss}_{\text{clean}})$$

**Intuition:** Apply Gaussian blur and additive noise to the image, then measure how much the loss increases. A memorized image should cause a larger loss increase (the model is tuned to the exact pixels), while a generalized image should be more robust to perturbation.

**Why it is weak (AUC = 0.5203):** SmolVLM's ViT vision encoder is inherently robust to Gaussian noise. The loss change is tiny for all samples (std = 0.015), providing almost no signal.

------------------------------------------------------------------------

## Method 2: Dual-Model Neighborhood Attack (Milestone 5)

Source: `milestone5/src/neighborhood_vlm.py`

Adapted from the text-only neighborhood attack (Milestone 3) to the multi-modal setting. This method generates perturbed "neighbors" of the input — in both text and image modalities — and measures how the loss changes under perturbation using both the finetuned and base models.

### Text Neighborhood Features

Perturb the caption by randomly dropping 10% of words (5 neighbors per sample). The image remains fixed.

| Feature | Description |
|------------------------------|------------------------------------------|
| `text_nb_gap_ft` | mean(perturbed losses) - original loss (finetuned model) |
| `text_nb_gap_base` | mean(perturbed losses) - original loss (base model) |
| `text_nb_std_ft` | std of perturbed losses (finetuned model) |
| `text_nb_relative_gap` | `text_nb_gap_ft` - `text_nb_gap_base` |

**Intuition:** If the model memorized the exact caption, any word drop should increase the loss. The relative gap compares how much more sensitive the finetuned model is versus the base model to the same perturbation.

### Image Neighborhood Features

Perturb the image using three strategies (cycling through Gaussian noise, patch masking, random crop+resize; 5 neighbors per sample). The text remains fixed.

| Feature | Description |
|------------------------------|------------------------------------------|
| `img_nb_gap_ft` | mean(perturbed losses) - original loss (finetuned model) |
| `img_nb_gap_base` | mean(perturbed losses) - original loss (base model) |
| `img_nb_std_ft` | std of perturbed losses (finetuned model) |
| `img_nb_relative_gap` | `img_nb_gap_ft` - `img_nb_gap_base` |

**Intuition:** If the model memorized the exact image, any pixel-level perturbation should increase the loss. The `img_nb_std_ft` feature (AUC = 0.5804) is the strongest individual M5 feature — a higher std indicates the model's sensitivity varies across perturbation types, which is a sign of overfitting to specific visual features.

### Cross-Modal Binding Features

| Feature           | Description                         |
|-------------------|-------------------------------------|
| `cross_modal_gap` | `text_nb_gap_ft` - `img_nb_gap_ft`  |
| `binding_score`   | `text_nb_gap_ft` \* `img_nb_gap_ft` |
| `total_nb_gap`    | `text_nb_gap_ft` + `img_nb_gap_ft`  |

**Intuition:** For a true member (joint memorization of image + caption), both text and image perturbations should increase the loss. The `binding_score` (product of both gaps) captures whether the model is sensitive to perturbations in **both** modalities simultaneously.

### Text Statistics

| Feature        | Description                          |
|----------------|--------------------------------------|
| `text_len`     | Number of words in the text          |
| `unique_ratio` | Ratio of unique words to total words |

### Base Loss Features

| Feature     | Description                                             |
|-------------|---------------------------------------------------------|
| `loss_ft`   | Finetuned model loss on the original (image, text) pair |
| `loss_base` | Base model loss on the original (image, text) pair      |
| `ref_diff`  | `loss_base` - `loss_ft`                                 |
| `ref_ratio` | `loss_ft` / `loss_base`                                 |

------------------------------------------------------------------------

## Combined Pipeline

### Why Combining Works

Milestone 6's `loss_ratio` (AUC = 0.79) is the dominant signal — it directly measures how much finetuning changed the model's confidence on a given pair. Milestone 5's neighborhood features are individually weak (AUC 0.50–0.58), but they capture **complementary** information:

-   `loss_ratio` captures **absolute memorization** (how much more confident the finetuned model is)
-   Neighborhood features capture **relative sensitivity** (how the model responds to perturbation)

These are orthogonal signals. A sample can have a moderate loss ratio but high perturbation sensitivity, or vice versa. The combination allows the classifier to identify members that either signal alone would miss.

### Feature Selection

Forward feature selection (5-fold CV on train) identified 14 features as optimal from the 20-feature set. However, the full 20-feature LR model performed better on validation (AUC 0.8486 vs 0.8225 for the selected subset), suggesting the additional features provide useful regularization at test time.

### Classifier Choice

Logistic Regression with `class_weight='balanced'` outperformed XGBoost on [TPR\@FPR](mailto:TPR@FPR){.email}=0.1 (0.388 vs 0.372) despite nearly identical AUC (0.849 vs 0.849). With only 20 features and 6000 training samples, the linear model generalizes better and produces better-calibrated probabilities in the low-FPR regime. XGBoost's additional complexity does not help when the feature space is small and the relationships are approximately linear.

------------------------------------------------------------------------

## Results

### Validation Set Performance (500 samples, 121 members / 379 non-members)

| Method | Model | \# Features | AUC | [TPR\@FPR](mailto:TPR@FPR){.email}=0.1 |
|----|----|----|----|----|
| M6 only | LR | 3 | 0.8154 | 0.3554 |
| M5 only | LR | 17 | 0.6305 | 0.1653 |
| M5 only | XGB | 17 | 0.6142 | 0.1570 |
| M6 + CLIP | LR | 7 | 0.8163 | 0.2727 |
| M6 + M5 + CLIP | LR | 24 | 0.8322 | 0.2727 |
| **M6 + M5 (final)** | **LR** | **20** | **0.8486** | **0.3884** |
| M6 + M5 (final) | XGB | 20 | 0.8490 | 0.3719 |

### Per-Feature Univariate AUC (Validation Set)

| Feature                  | Source | AUC        | Direction           |
|--------------------------|--------|------------|---------------------|
| `loss_ratio`             | M6     | **0.7929** | \- (lower = member) |
| `img_nb_std_ft`          | M5     | 0.5804     | \-                  |
| `text_nb_relative_gap`   | M5     | 0.5395     | \-                  |
| `loss_base`              | M5     | 0.5329     | \-                  |
| `text_nb_gap_base`       | M5     | 0.5292     | \+                  |
| `ref_diff`               | M5     | 0.5283     | \-                  |
| `corruption_sensitivity` | M6     | 0.5203     | \-                  |
| `img_nb_gap_base`        | M5     | 0.5206     | \+                  |
| `img_nb_relative_gap`    | M5     | 0.5204     | \-                  |
| `caption_contrast`       | M6     | 0.5085     | \+                  |

### 5-Fold Cross-Validation AUC (Train Set, 6000 samples)

| Method           | CV AUC     |
|------------------|------------|
| M6 only (LR)     | 0.7839     |
| M5 only (LR)     | 0.6042     |
| **M6 + M5 (LR)** | **0.8114** |

------------------------------------------------------------------------

## Submission Details

-   **File:** `milestone6/data/submission.csv`
-   **Format:** `id, is_member` (probability score)
-   **Rows:** 6000 test samples
-   **Model:** Logistic Regression, `class_weight='balanced'`, `max_iter=1000`
-   **Features:** 20 (3 from M6 + 17 from M5)
-   **Preprocessing:** StandardScaler (fit on train, transform on test)
-   **Score distribution:** mean = 0.4207, std = 0.2902
