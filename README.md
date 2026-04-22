# Membership Inference Attacks on LLMs and VLMs

Research project implementing and evaluating membership inference attack (MIA) methods against fine-tuned language models and vision-language models. MIA determines whether a given sample was used during model fine-tuning — a core technique in auditing ML privacy and memorization.

## Repository Structure

```
src/
├── llm/          # Text-only MIA methods (SmolLM2)
└── vlm/          # Vision-Language MIA methods (SmolVLM)
docs/
├── data/         # Data inspection notes
├── methods/      # Method write-ups
├── slides/       # Project presentation
└── paper/        # Research paper
```

---

## LLM Attacks — `src/llm/`

**Target model:** `UBC-SLIME/colx_531_smollm2-135m` (SmolLM2-135M fine-tuned on medical discharge summaries)  
**Dataset:** `UBC-SLIME/colx_531_group_project` — 50k train / 10k dev / 15k test clinical notes  
**Evaluation metric:** AUC (ROC), TPR@FPR=0.1

| Script | Method | Val AUC | Val TPR@FPR=0.1 |
|--------|--------|:-------:|:---------------:|
| `baseline.py` | Raw Loss (Baseline) | 0.5839 | 0.1552 |
| `reference_model.py` | Reference Model (Loss Ratio) | 0.6701 | 0.2214 |
| `min_k_prob.py` | Min-K% Prob (K=20%) | 0.6166 | 0.1682 |
| `casing_attack.py` | Casing Attack | 0.5887 | 0.1566 |
| **`neighborhood_attack.py`** | **Neighborhood Attack (GBC)** | **0.9071** | **0.6966** |

---

## VLM Attacks — `src/vlm/`

**Target model:** `UBC-SLIME/colx_585_vlm` (SmolVLM-256M-Instruct fine-tuned on image-caption pairs)  
**Dataset:** `UBC-SLIME/colx585_group_project_data` — 6k train / 1.2k val / 6k test image-caption pairs  
**Evaluation metric:** AUC (ROC), TPR@FPR=0.1

| Script | Method | Val AUC | Val TPR@FPR=0.1 |
|--------|--------|:-------:|:---------------:|
| `baseline.py` | Raw Loss (Baseline) | 0.5008 | 0.1200 |
| `min_k_prob.py` | Multi-Feature Min-K% Prob | 0.5127 | 0.0800 |
| `mia_pipeline.py` | Lightweight Complexity Calibration | 0.5605 | 0.1800 |
| `neighborhood_vlm.py` | Neighborhood Attack | 0.5988 | 0.1867 |
| `m4i_clip_fast.py` | M⁴I-CLIP Attack | 0.8093 | 0.3000 |
| `mia_LiRA.py` | Hybrid MIA + LiRA | 0.8061 | 0.2633 |
| **`mia_attack.py` + `neighborhood_vlm.py`** | **Neighborhood + MIA Combined** | **0.8486** | **0.3884** |

---

## Key Findings

- **Neighborhood Attack dominates the LLM track** — GradientBoosting on a 16-feature dual-model vector achieves AUC 0.91 and TPR@FPR=0.1 of 0.70, far ahead of simple loss-based methods (AUC ≈ 0.58–0.67).
- **CLIP image-text similarity is the strongest VLM signal** — `clip_img_text_sim` alone accounts for 48% of XGBoost feature importance. Members show higher image-text alignment in CLIP space.
- **VLM generation adds minimal signal** — M⁴I-CLIP Full (with VLM generation) barely outperforms CLIP-only (Δ AUC = +0.00023 on Kaggle), suggesting the membership signal is primarily in data characteristics rather than model outputs.
- **Combining orthogonal signals helps in VLM** — Loss ratio features (AUC 0.82) and neighborhood perturbation features (AUC 0.60) are complementary; their combination raises TPR@FPR=0.1 from 0.355 → 0.388.

---

## Setup

```bash
pip install -r requirements.txt
```

## References

- Hu et al. (2022). *M⁴I: Multi-modal Models Membership Inference*. NeurIPS 2022.
- Carlini et al. (2022). *Membership Inference Attacks From First Principles*. IEEE S&P 2022.
- Li et al. (2024). *Membership Inference Attacks against Large Vision-Language Models*. NeurIPS 2024.
- Hu et al. (2025). *Membership Inference Attacks Against Vision-Language Models*. USENIX Security 2025.
- Shi et al. (2024). *Detecting Pretraining Data from Large Language Models*. ICLR 2024. (Min-K% Prob)
