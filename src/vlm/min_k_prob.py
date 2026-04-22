"""
File: mia_improved.py
Description:
    Improved Membership Inference Attack on SmolVLM.
    
    Baseline used only aggregate cross-entropy loss as a single feature.
    This script extracts multiple per-token statistics as MIA signals:
    
    1. Loss (mean per-token CE loss) — same as baseline
    2. Perplexity (exp(loss)) — nonlinear transform emphasising high-loss outliers
    3. Min-K% Prob — average log-probability of the lowest-k% tokens;
       members have higher values because the model memorised even "hard" tokens
    4. Max token loss — the single worst token's loss; lower for members
    5. Std of token losses — members have more uniform (lower std) token losses
    6. Zlib ratio — loss / zlib_compressed_length; normalises by text complexity
    
    All features are combined into a Logistic Regression classifier.
"""

import torch
import transformers
import datasets
from datasets import load_dataset
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from PIL import Image
import io
import os
import json
import zlib
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"datasets: {datasets.__version__}")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and processor
model_id = "UBC-SLIME/colx_585_vlm"
dataset_id = "UBC-SLIME/colx585_group_project_data"

processor = AutoProcessor.from_pretrained(model_id)
processor.image_processor.do_image_splitting = False
processor.image_processor.size = {"longest_edge": 512}
processor.image_processor.max_image_size = {"longest_edge": 512}

model = SmolVLMForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    _attn_implementation="sdpa",
    trust_remote_code=True,
).to(device)
model.eval()
print("Model loaded successfully.")

# Configuration
MIN_K_PERCENT = 20  # Bottom 20% of token probabilities


# Feature Extraction

def extract_image(example):
    """Extract PIL image from a dataset example."""
    if isinstance(example["image"], dict) and "bytes" in example["image"]:
        try:
            return Image.open(io.BytesIO(example["image"]["bytes"])).convert("RGB")
        except Exception:
            import base64
            return Image.open(
                io.BytesIO(base64.b64decode(example["image"]["bytes"]))
            ).convert("RGB")
    else:
        try:
            return example["image"].convert("RGB")
        except Exception:
            return Image.new("RGB", (512, 512))


def compute_features(example, processor, model, device):
    """
    Compute multiple MIA features from a single example.
    
    Returns a dict with: loss, perplexity, min_k_prob, max_token_loss,
    std_token_loss, zlib_ratio, num_tokens
    """
    image = extract_image(example)

    parts = example["text"].split("\n", 1)
    user_text = parts[0]
    assistant_text = parts[1] if len(parts) > 1 else ""

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    if device == "cuda":
        inputs = {
            k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
            for k, v in inputs.items()
        }

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Per-token log probabilities
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].float()  # cast to float for stable softmax
    shift_labels = labels[:, 1:]

    # Log probabilities of the correct tokens
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (1, seq_len-1)

    # Mask out positions where label is -100 (non-target tokens)
    mask = shift_labels != -100
    token_log_probs = token_log_probs[mask]  # (num_target_tokens,)

    if token_log_probs.numel() == 0:
        return None  # Skip if no valid tokens

    token_losses = -token_log_probs  # Per-token CE loss (positive values)

    # Feature 1: Mean loss (same as baseline)
    mean_loss = token_losses.mean().item()

    # Feature 2: Perplexity 
    perplexity = np.exp(mean_loss)

    # Feature 3: Min-K% Prob 
    # Average log-prob of the bottom K% tokens (lowest probability tokens)
    # Members should have HIGHER min-k values (model memorised hard tokens)
    k = max(1, int(len(token_log_probs) * MIN_K_PERCENT / 100))
    sorted_log_probs, _ = token_log_probs.sort()
    min_k_prob = sorted_log_probs[:k].mean().item()

    # Feature 4: Max token loss
    max_token_loss = token_losses.max().item()

    # Feature 5: Std of token losses
    std_token_loss = token_losses.std().item() if len(token_losses) > 1 else 0.0

    # Feature 6: Zlib ratio
    # Normalises loss by text compressibility (intrinsic complexity)
    text_bytes = example["text"].encode("utf-8")
    zlib_len = len(zlib.compress(text_bytes))
    zlib_ratio = mean_loss / (zlib_len + 1e-8)

    # Feature 7: Token count
    num_tokens = int(mask.sum().item())

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "min_k_prob": min_k_prob,
        "max_token_loss": max_token_loss,
        "std_token_loss": std_token_loss,
        "zlib_ratio": zlib_ratio,
        "num_tokens": num_tokens,
    }


# Data Processing 

def process_split(split_name, max_samples=None, cache_path=None, has_labels=True):
    """Process a dataset split and return a list of feature dicts."""
    # Check cache
    if cache_path and os.path.exists(cache_path) and not max_samples:
        print(f"Loading cached features from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    print(f"Computing features for {split_name} split...")
    ds = load_dataset(dataset_id, split=split_name, streaming=True)

    data = []
    default_label = 1 if split_name == "train" else 0
    estimate_total = {"train": 6000, "validation": 1200, "test": 6000}
    total = max_samples if max_samples else estimate_total.get(split_name, 1000)

    pbar = tqdm(total=total, desc=split_name.capitalize())
    for i, example in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        try:
            features = compute_features(example, processor, model, device)
            if features is None:
                continue
            features["id"] = example["id"]
            if has_labels:
                features["is_member"] = example.get("is_member", default_label)
            data.append(features)
        except Exception as e:
            print(f"Error processing {split_name} sample {example.get('id', i)}: {e}")
        pbar.update(1)
    pbar.close()

    # Cache results
    if cache_path and not max_samples:
        with open(cache_path, "w") as f:
            json.dump(data, f)

    return data


#  Main

FEATURE_COLS = [
    "loss", "perplexity", "min_k_prob",
    "max_token_loss", "std_token_loss", "zlib_ratio", "num_tokens",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run on small subset")
    args = parser.parse_args()

    max_samples = 100 if args.debug else None
    cache_suffix = "" if not args.debug else "_debug"

    # Extract features
    train_data = process_split(
        "train", max_samples,
        cache_path=f"train_features_improved{cache_suffix}.json",
    )
    val_data = process_split(
        "validation", max_samples,
        cache_path=f"val_features_improved{cache_suffix}.json",
    )

    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)

    if df_train.empty:
        print("No training data. Exiting.")
        return

    # Train classifier 
    X_train = df_train[FEATURE_COLS]
    y_train = df_train["is_member"]

    # Standardize features for stable LR training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # Print feature importances (LR coefficients)
    print("\nFeature importances (LR coefficients):")
    for name, coef in zip(FEATURE_COLS, clf.coef_[0]):
        print(f"  {name:<18s}: {coef:+.4f}")

    # Evaluate on validation
    if not df_val.empty and "is_member" in df_val.columns:
        X_val = df_val[FEATURE_COLS]
        X_val_scaled = scaler.transform(X_val)
        y_val = df_val["is_member"]

        y_val_prob = clf.predict_proba(X_val_scaled)[:, 1]

        try:
            auc = roc_auc_score(y_val, y_val_prob)
            fpr, tpr, _ = roc_curve(y_val, y_val_prob)
            tpr_at_10 = tpr[fpr <= 0.1][-1] if len(tpr[fpr <= 0.1]) > 0 else 0.0
            print(f"\nValidation ROC AUC:    {auc:.4f}")
            print(f"Validation TPR@FPR=0.1: {tpr_at_10:.4f}")

            # Also report individual feature AUCs for comparison
            print("\nPer-feature AUC (univariate, no LR):")
            for col in FEATURE_COLS:
                try:
                    # For loss-like features, higher = non-member → flip sign
                    feat_auc = roc_auc_score(y_val, -df_val[col])
                    feat_auc = max(feat_auc, 1 - feat_auc)  # Take better direction
                    print(f"  {col:<18s}: {feat_auc:.4f}")
                except ValueError:
                    print(f"  {col:<18s}: N/A")
        except ValueError as e:
            print(f"Could not calculate metrics: {e}")

    # Predict on test
    test_data = process_split(
        "test", max_samples,
        cache_path=f"test_features_improved{cache_suffix}.json",
        has_labels=False,
    )
    df_test = pd.DataFrame(test_data)
    if df_test.empty:
        print("No test data. Exiting.")
        return

    X_test = df_test[FEATURE_COLS]
    X_test_scaled = scaler.transform(X_test)
    df_test["is_member"] = clf.predict_proba(X_test_scaled)[:, 1]

    # Export
    output_dir = os.path.join("milestone4", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "improved_submission.csv")
    df_test[["id", "is_member"]].to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
