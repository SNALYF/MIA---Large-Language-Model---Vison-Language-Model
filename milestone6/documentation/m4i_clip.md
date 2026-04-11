# M⁴I-CLIP: Adapted Multi-modal Membership Inference Attack

## Overview

This method adapts the two core attack strategies from the **M⁴I** paper (Hu et al., NeurIPS 2022) from last week's literature review— *Metric-Based* and *Feature-Based* membership inference — to the Kaggle VLM MIA competition. Instead of using the paper's legacy architectures (ResNet-152 + LSTM decoder with shadow models), we modernize both approaches using **CLIP** as the cross-modal feature extractor and the target **SmolVLM** as the generative model.

### Reference Paper
> Hu, T., Wang, R., Jia, J., & Li, B. (2022). *Membership Inference of Multi-modal Multi-model Learning: A Comprehensive Study*. NeurIPS 2022.
> GitHub: https://github.com/MultimodalMI/Multimodal-membership-inference

---

## Methodology

### Method 1: Modernized Metric-Based M⁴I (MB-M⁴I)

**Original idea (paper):** Generate a caption from the target model for a given image, then compare it to the ground-truth caption using ROUGE/BLEU. Members produce more similar outputs because the model memorized them.

**Our adaptation:**
1. Feed each image into the finetuned SmolVLM to generate a text response (greedy decoding)
2. Compare the generated text to the ground-truth text using:
   - **ROUGE-L F1** — longest common subsequence overlap
   - **Token overlap ratio** — bag-of-words Jaccard similarity

**Key difference from original:** We dropped BLEU in favor of simpler token overlap, as both BLEU and ROUGE capture n-gram overlap redundantly.

### Method 2: Modernized Feature-Based M⁴I (FB-M⁴I)

**Original idea (paper):** Project the image and text into a shared embedding space using a custom Multi-modal Feature Extractor (MFE trained on ResNet-152 + text-CNN), then measure Euclidean distance. Members have tighter image-text alignment because the model overfitted to those pairs.

**Our adaptation:**
1. Use **OpenAI CLIP (ViT-B/32)** as the off-the-shelf cross-modal feature extractor (replacing the custom MFE)
2. For each sample, compute cosine similarities between:
   - Image embedding ↔ ground-truth text embedding (`clip_img_text_sim`)
   - Image embedding ↔ generated text embedding (`clip_img_gen_sim`)
   - Ground-truth text embedding ↔ generated text embedding (`clip_text_gen_sim`)
   - The gap: `clip_sim_gap = clip_img_gen_sim - clip_img_text_sim`

**Key difference from original:** Euclidean distance in a custom learned space → cosine similarity in CLIP space (stronger, general-purpose alignment metric).

### Meta-Classifier

All features are combined and fed into an **XGBoost** gradient-boosted classifier (consistent with the milestone5 pipeline).

---

## Feature Set

### Full M⁴I-CLIP Attack (`m4i_clip_attack.py`)
Requires VLM text generation — slower but captures generation-based signals.

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `rouge_l` | MB-M⁴I | ROUGE-L F1 between generated and ground-truth text |
| 2 | `token_overlap` | MB-M⁴I | Jaccard token overlap ratio |
| 3 | `clip_img_text_sim` | FB-M⁴I | CLIP cosine similarity: image ↔ ground-truth text |
| 4 | `clip_img_gen_sim` | FB-M⁴I | CLIP cosine similarity: image ↔ generated text |
| 5 | `clip_text_gen_sim` | FB-M⁴I | CLIP cosine similarity: ground-truth ↔ generated text |
| 6 | `clip_sim_gap` | FB-M⁴I | Gap between generated and ground-truth alignment |
| 7 | `gen_length` | Auxiliary | Word count of generated text |

### Fast CLIP-Only Attack (`m4i_clip_fast.py`)
No VLM generation needed — runs in minutes. Merges with milestone5 loss-based features.

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `clip_img_text_sim` | FB-M⁴I | CLIP cosine similarity: image ↔ ground-truth text |
| 2 | `clip_img_question_sim` | FB-M⁴I | CLIP cosine similarity: image ↔ question text |
| 3 | `clip_text_complexity` | FB-M⁴I | CLIP text embedding norm (complexity proxy) |
| 4–10 | `loss`, `perplexity`, `min_k_prob`, etc. | Milestone 5 | Pre-computed loss-based statistics |

---

## Results

### Kaggle Leaderboard (AUC)

| Method | Kaggle AUC | Notes |
|--------|-----------|-------|
| **M⁴I-CLIP Full** (with VLM generation) | **0.80560** | 7 features, ~8 hrs on MPS |
| **M⁴I-CLIP Fast** (CLIP-only + M5 loss) | **0.80537** | 10 features, ~15 min on MPS |

### Validation Set Metrics

| Method | Val AUC | Val TPR@FPR=0.1 |
|--------|---------|-----------------|
| M⁴I-CLIP Full | 0.8487 (debug) | 0.3333 (debug) |
| M⁴I-CLIP Fast | 0.8093 | 0.3000 |

### Feature Analysis

#### Per-Feature Univariate AUC (M⁴I-CLIP Full, debug set)

| Feature | AUC | Direction |
|---------|-----|-----------|
| `clip_sim_gap` | **0.8991** | − (members have more negative gap) |
| `clip_img_text_sim` | **0.8553** | + (members have higher similarity) |
| `clip_img_gen_sim` | 0.6162 | − |
| `clip_text_gen_sim` | 0.6162 | + |
| `gen_length` | 0.5844 | + |
| `token_overlap` | 0.5285 | + |
| `rouge_l` | 0.5230 | + |

#### Feature Importances (XGBoost, M⁴I-CLIP Fast, full dataset)

| Feature | Importance |
|---------|-----------|
| `clip_img_text_sim` | **0.4836** |
| `min_k_prob` | 0.0642 |
| `clip_img_question_sim` | 0.0600 |
| `clip_text_complexity` | 0.0580 |
| All others | < 0.06 each |

### Key Findings

1. **`clip_img_text_sim` dominates** — CLIP cosine similarity between image and ground-truth text is the single most important feature (48% of XGBoost importance). Members have higher image-text alignment in CLIP space.

2. **VLM generation adds minimal value** — The full attack (0.80560) barely outperforms CLIP-only (0.80537), a Δ of just +0.00023 AUC. This suggests the membership signal is primarily in the **data characteristics** (how well-aligned the image-text pair is) rather than in **model behavior** (what the VLM generates).

3. **Milestone5 loss features are weak alone** — All loss-based features (perplexity, min-k-prob, etc.) have univariate AUC around 0.50-0.51, meaning they contribute minimally to discrimination.

4. **`clip_sim_gap` is theoretically the most interesting** — At 0.90 AUC on the debug set, it captures whether the model's generated text is *more* aligned to the image than the reference text, which is a direct signature of memorization.

---

## Architecture Comparison

| Aspect | Original M⁴I Paper | Our Adaptation |
|--------|-------------------|----------------|
| Target model | ResNet-152 + LSTM decoder | SmolVLM-256M-Instruct (VLM) |
| Shadow model | Required (separate training) | **Not needed** |
| MB features | ROUGE, BLEU | ROUGE-L, Token Overlap |
| FB extractor | Custom MFE (ResNet-152 + text-CNN) | **CLIP ViT-B/32** (off-the-shelf) |
| FB metric | Euclidean distance in learned space | **Cosine similarity in CLIP space** |
| Classifier | SVM (linear kernel) | **XGBoost** (gradient boosting) |
| Dataset | Flickr8k / COCO 2017 | UBC-SLIME/colx585_group_project_data (HuggingFace) |
| Data loading | Local files | HuggingFace streaming |

---

## Usage

### Full Attack (with VLM generation)
```bash
cd milestone6/src

# Debug run (~50 samples, ~3 min)
python m4i_clip_attack.py --debug

# Full run (~13,200 samples, ~6-8 hrs on MPS, ~1-2 hrs on CUDA)
python m4i_clip_attack.py

# Force CUDA (for Colab / RTX GPU)
python m4i_clip_attack.py --device cuda

# Re-extract features (ignore cache)
python m4i_clip_attack.py --recompute
```

### Fast CLIP-Only Attack (no VLM needed)
```bash
# Full run (~13,200 samples, ~15 min)
python m4i_clip_fast.py

# Force CUDA
python m4i_clip_fast.py --device cuda
```

### Output Files
- `output/m4i_clip_submission.csv` — Kaggle submission (full attack)
- `output/m4i_clip_attack_submission.csv` — Kaggle submission (fast attack)
- `output/m4i_clip_features.csv` — Detailed test features
- `src/m4i_cache/*.json` — Cached features per split

---

## Dependencies

```
pip install rouge-score open_clip_torch xgboost scikit-learn datasets transformers accelerate Pillow
```
