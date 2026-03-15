# TFC - COLX 531 Group Project

## Project Overview

This repository contains the group project for COLX 531. The task involves membership inference attack on finetuned language models.

## Repository Structure

```
TFC/
├── README.md               # This file
├── documentation/          # Documentation for the project
│   ├── milestone1/
│   │   ├── teamwork_contract.md
│
├── milestone1/             # Milestone 1 deliverables
│   ├── teamwork_contract.md
│   ├── data_inspection.md

├── src/                    # Source code (shared across milestones)
│   └── milestone1/
│      └── data_insepction.py
└── lab3.md                 # Lab 3 instructions
```

## Branch Strategy

- `main` — The stable branch. **Never push directly to main.** All changes must be merged via pull requests reviewed by at least one other team member.
- Individual branches (e.g., `tianhao`, etc.) — Each team member works on their own branch and creates PRs to merge into `main`.

## Milestones

| Milestone   | Folder        | Status      |
|-------------|---------------|-------------|
| Milestone 1 | `milestone1/` | Finished |
| Milestone 2 | `milestone2/` | Finished |

## Getting Started

### Load the Dataset

```python
from datasets import load_dataset
dataset = load_dataset("UBC-SLIME/colx_531_group_project")
```

### Load the Finetuned Models

```python
from transformers import AutoModelForCausalLM

lm1 = AutoModelForCausalLM.from_pretrained("UBC-SLIME/colx_531_smollm2-135m")
lm2 = AutoModelForCausalLM.from_pretrained("UBC-SLIME/colx_531_smollm2-360m")
```

## Team Members

- Member 1: Tianhao Cao
- Member 2: Yusen Huang
- Member 3: Marco Wang
- Member 4: Darwin Zhang
