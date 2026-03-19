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
├── milestone1/                        # Milestone 1 deliverables
│   ├── documentation/
│   │   ├── data_inspection.md
│   │   ├── baseline_submission.csv
│   │   └── milestone1_group_report.md
│   └── src/
│       ├── data_inspection.py
│       └── baseline.py
├── milestone2/                        # Milestone 2 deliverables
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
└── milestone3/                        # Milestone 3 deliverables
    ├── document/
    │   └── milestone3_group_report.md
    └── src/
        └── neighborhood_attack.py
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

## Team Members

- Member 1: Tianhao Cao
- Member 2: Yusen Huang
- Member 3: Marco Wang
- Member 4: Darwin Zhang
