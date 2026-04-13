# TFC - COLX 531 Group Project

## Project Overview

This repository contains the group project for COLX 531. The task involves membership inference attack on finetuned language models.

## Repository Structure

```
TFC/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── documentation/                     # Project-level documentation
│   └── team_contract.md
├── milestone1/                        # Milestone 1: Data inspection & baseline
│   ├── documentation/
│   │   ├── data_inspection.md
│   │   ├── baseline_submission.csv
│   │   └── milestone1_group_report.md
│   └── src/
│       ├── data_inspection.py
│       └── baseline.py
├── milestone2/                        # Milestone 2: Text-only MIA attacks
│   ├── documentation/
│   │   └── milestone2_group_report.md
│   └── src/
│       ├── casing_attack.py
│       ├── min_k_prob.py
│       ├── metric_threshold_attack.py
│       ├── reference_model.py
│       └── milestone2/               # Submission outputs
│           ├── casing_attack_submission.csv
│           ├── metric_threshold_submission.csv
│           ├── metric_threshold_all_scores.csv
│           ├── neighborhood_attack_submission.csv
│           ├── neighborhood_attack_details.csv
│           ├── camia_submission.csv
│           └── camia_all_signals.csv
├── milestone3/                        # Milestone 3: Text neighborhood attack
│   ├── document/
│   │   ├── milestone3_group_report.md
│   │   └── Team1.pdf
│   └── src/
│       └── neighborhood_attack.py
├── milestone4/                        # Milestone 4: VLM data inspection & baseline
│   ├── documentation/
│   │   ├── baseline.md
│   │   ├── data_inspection.md
│   │   ├── milestone4_group_report.md
│   │   └── milestone4_group_report.pdf
│   └── src/
│       ├── baseline.py
│       ├── data_inspection.py
│       ├── train_features.json
│       ├── val_features.json
│       └── test_features.json
├── milestone5/                        # Milestone 5: VLM neighborhood attack
│   ├── documentation/
│   │   ├── milestone5_group_report.md
│   │   └── milestone_5_group_report.pdf
│   └── src/
│       ├── mia_pipeline.py
│       ├── min_k_prob.py
│       ├── neighborhood_vlm.py
│       ├── train_features_improved.json
│       ├── val_features_improved.json
│       └── test_features_improved.json
└── milestone6/                        # Milestone 6: Combined MIA + neighborhood attack
    ├── documentation/
    │   ├── m4i_clip.md
    │   ├── neighborhood_mia_approach.md
    │   └── usenixsecurity25-hu-yuke.pdf
    └── src/
        ├── mia_attack.py             # Three-layer MIA (loss ratio, caption contrast, corruption)
        ├── m4i_clip_attack.py         # M⁴I-adapted attack (CLIP + metric-based)
        └── m4i_clip_fast.py           # Fast CLIP-only attack
```

## Branch Strategy

- `main` — The stable branch. **Never push directly to main.** All changes must be merged via pull requests reviewed by at least one other team member.
- Individual branches (e.g., `tianhao`, etc.) — Each team member works on their own branch and creates PRs to merge into `main`.

## Milestones

| Milestone   | Folder        | Status      |
|-------------|---------------|-------------|
| Milestone 1 | `milestone1/` | Finished |
| Milestone 2 | `milestone2/` | Finished |
| Milestone 3 | `milestone3/` | Finished |
| Milestone 4 | `milestone4/` | Finished |
| Milestone 5 | `milestone5/` | Finished |
| Milestone 6 | `milestone6/` | Finished |

## Team Members

- Member 1: Tianhao Cao
- Member 2: Yusen Huang
- Member 3: Marco Wang
- Member 4: Darwin Zhang
