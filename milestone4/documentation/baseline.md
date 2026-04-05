---
title: "baseline.md"
author: "Tianhao Cao"
date: "2026-03-29"
disclaimer: "This document is written with the help of Gemini 3.1 Pro"
---

# Baseline Implementation Documentation

This document describes the implementation details of the `baseline.py` script. The script provides a baseline Membership Inference Attack (MIA) against a Vision-Language Model (VLM).

## 1. Overview
The primary goal of the script is to differentiate whether a given data sample belongs to the model's training set (member) or not (non-member) by calculating the model's loss on the sample. It loads a fine-tuned VLM model, extracts loss features for all data samples, and trains a Logistic Regression classifier on these features to output membership predictions on the test set.

## 2. Core Dependencies and Setup
- **Model**: `UBC-SLIME/colx_585_vlm` (built on the `SmolVLMForConditionalGeneration` architecture, utilizing SDPA attention for acceleration).
- **Dataset**: `UBC-SLIME/colx585_group_project_data`.
- **Compute Device**: Automatically detects CUDA availability. If available, it uses the GPU with `bfloat16` precision; otherwise, it falls back to the CPU with `float32` precision.

## 3. Core Modules Breakdown

### 3.1 Sample Loss Computation (`compute_loss`)
This is the core feature extraction function of the baseline method:
- **Image Processing**: Extracts image bytes from the sample and converts them to an RGB `PIL.Image`. It has fallback mechanisms using Base64 decoding or generating a dummy blank image if the image data format is problematic.
- **Text Processing**: Splits the dataset text into User instructions and Assistant responses. It then formats these using the model processor's `apply_chat_template` into a conversational format.
- **Calculation Mechanism**: Within a `torch.no_grad()` context, the forward pass calculates the cross-entropy loss of predicting the `assistant` text. This loss serves as the sole signal for inferring whether the sample is a training member.

### 3.2 Data Processing and Feature Caching
To minimize the significant overhead of repeated inference, the script incorporates a caching mechanism for three data splits, saving the extracted features into JSON files:
1. **Train Split (~6000 samples)**:
   - Extracts `loss` and sets the `is_member` label to `1`.
   - Results are cached in `train_features.json`.
2. **Validation Split (~1200 samples)**:
   - Extracts `loss` alongside the target `is_member` for performance evaluation.
   - Results are cached in `val_features.json`.
3. **Test Split (~6000 samples)**:
   - Extracts only the `loss` for final inference.
   - Results are cached in `test_features.json`.

> **Note**: The script supports a `--debug` argument. When enabled, it processes a maximum of 100 samples per split and skips all caching (both reading and writing), which is useful for quick debugging.

### 3.3 Model Training and Evaluation
- **Logistic Regression Training**: Based on the `loss` extracted from the `train_features`, the script trains an `sklearn` Logistic Regression model. It uses the `class_weight='balanced'` parameter to automatically handle potential positive/negative class imbalances.
- **Metrics Evaluation**: The model's capabilities are evaluated on the Validation split, outputting two key attack metrics:
  - **ROC AUC Score**: Measures the model's overall ability to correctly rank and classify samples.
  - **TPR@FPR=0.1**: The False Positive Rate is held at 10%, and the True Positive Rate is reported (i.e., the proportion of actual members successfully identified).

### 3.4 Final Prediction and Output
After training, the classifier is applied to the test set (`test_features.json`).
- **Score Generation**: The prediction probabilities `predict_proba()[:, 1]` from the Logistic Regression model are used as continuous scores indicating the sample's likelihood of being a member.
- **Output Format**: The parsed `id` and the calculated `is_member` probabilities are outputted to the `milestone4/output/baseline_submission.csv` file, which serves as the final submission format for evaluation.

### 4. Equipment and Configuration:
I'm using a windows laptop with RTX 4070 Laptop GPU, 8GB VRAM, and 32GB RAM, with i9-13900HX CPU to run the script, the script takes around 15 minutes to run on my laptop.