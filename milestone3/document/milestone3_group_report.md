# Milestone 3 Group Report

**Team:** TFC **Members:** Tianhao Cao, Yusen Huang, Marco Wang, Darwin Zhang

------------------------------------------------------------------------

## Model 1: Improved Neighborhood Attack (Dual-Model Feature Fusion)

### Model Description

We implemented an **Improved Neighborhood Attack** that combines dual-model reference signals with neighborhood perturbation analysis. The attack leverages all four available models — two finetuned models (`colx_531_smollm2-135m`, `colx_531_smollm2-360m`) and their corresponding base checkpoints (`SmolLM2-135M`, `SmolLM2-360M`) — to extract a rich 16-dimensional feature vector per sample. Instead of using a computationally expensive BERT mask-and-fill strategy for neighbor generation, we adopt a fast **token-drop perturbation**, randomly removing 10% of words to produce 5 neighbors per sample. These features are then combined using a **GradientBoosting classifier** trained on labeled train/validation data.

### Implementation

We implemented this method in `milestone3/src/neighborhood_attack.py`. The 16 extracted features fall into five categories:

- **Finetuned model losses**: per-sample loss from both finetuned models.
- **Base model losses**: per-sample loss from both base (pre-trained) models.
- **Reference ratios**: the difference (`loss_base − loss_finetuned`) and ratio (`loss_finetuned / loss_base`) for each model pair, isolating the memorization signal from inherent text difficulty.
- **Neighborhood gaps**: the difference between mean neighbor loss and original loss on each finetuned model, plus the standard deviation of neighbor losses.
- **Cross-model & text features**: the loss difference between the two finetuned models (`loss_diff_ft`), the gap difference across models (`gap_diff`), text length, and lexical diversity (unique word ratio).

A `GradientBoostingClassifier` (300 estimators, max depth 4, learning rate 0.05) is trained on standardized features with train-set labels and evaluated on the validation split.

### Results

We performed local evaluation on the `validation` split:

| Method | AUC | TPR@FPR=0.1 | Accuracy |
|:---|:--:|:--:|:--:|
| Baseline (Raw Loss) | 0.5839 | 0.1552 | 0.5622 |
| Improved Neighborhood Attack | **0.9071** | **0.6966** | **0.8248** |

Feature importance analysis revealed the top-3 signals accounting for 86% of total importance:

| Feature | Importance | Description |
|:---|:--:|:---|
| `loss_diff_ft` | 33.1% | Loss difference between two finetuned models |
| `ref_diff_2` | 30.0% | Base–finetuned loss gap (360M pair) |
| `ref_ratio_2` | 23.1% | Finetuned/base loss ratio (360M pair) |

### Discussions

The dramatic improvement (AUC: 0.58 → 0.91) is primarily driven by **cross-model reference signals** rather than neighborhood perturbation itself. The most informative feature, `loss_diff_ft`, captures systematic differences in how two finetuned models of different capacities memorize training data — members exhibit a characteristic loss gap between the 135M and 360M models that non-members do not. The 360M model's reference ratio (`ref_diff_2`, `ref_ratio_2`) contributed over 53% of the signal, suggesting that larger models produce stronger memorization signatures. Meanwhile, classical neighborhood features (`nb_gap`) contributed minimally (~0.5%), confirming that for finetuned (as opposed to pre-trained) LLMs, distributional-level memorization outweighs verbatim memorization of exact token sequences.
------------------------------------------------------------------------

## Contributions

### Milestone 3

| Member       | Contributions     | Percentage |
|--------------|-------------------|------------|
| Tianhao Cao  | Methods           | 25%        |
| Yusen Huang  | Methods           | 25%        |
| Marco Wang   | Methods           | 25%        |
| Darwin Zhang | Literature Review | 25%        |

**Total:** 100%
