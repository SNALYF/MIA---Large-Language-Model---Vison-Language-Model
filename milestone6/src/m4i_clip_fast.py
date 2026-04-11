"""
M⁴I-Adapted MIA: CLIP-Only Fast Submission
============================================

Extracts CLIP-based features for the full test set WITHOUT needing
VLM text generation (which is the bottleneck in the full pipeline).

Key insight from the debug run:
  - clip_img_text_sim alone had AUC 0.8553
  - This feature only needs the IMAGE and the GROUND-TRUTH TEXT
  - No VLM generation needed!

This script also merges the milestone5 loss-based features for
an even stronger combined classifier.

Usage:
    python m4i_clip_fast.py          # full run 
    python m4i_clip_fast.py --debug   # small subset
    python m4i_clip_fast.py --device cuda # use cuda
"""

import os
import io
import json
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

import open_clip
from xgboost import XGBClassifier


# Configuration 
DATASET_ID = "UBC-SLIME/colx585_group_project_data"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4i_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")

# Milestone 5 cached features (loss-based signals)
M5_FEATURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "milestone5", "src"
)

# CLIP-only features (no VLM generation needed)
CLIP_FEATURE_COLS = [
    "clip_img_text_sim",      # CLIP cosine(image, ground-truth text)
    "clip_img_question_sim",  # CLIP cosine(image, question only)
    "clip_text_complexity",   # Text embedding norm (proxy for complexity)
]

# Milestone 5 features to merge
M5_FEATURE_COLS = [
    "loss", "perplexity", "min_k_prob",
    "max_token_loss", "std_token_loss", "zlib_ratio", "num_tokens",
]

# Combined feature set
ALL_FEATURE_COLS = CLIP_FEATURE_COLS + M5_FEATURE_COLS


# Helpers
def tpr_at_fpr(y_true, y_score, target_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    return tpr[idx[-1]] if len(idx) > 0 else 0.0


def extract_image(example):
    img = example["image"]
    if isinstance(img, dict) and "bytes" in img:
        try:
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        except Exception:
            import base64
            return Image.open(io.BytesIO(base64.b64decode(img["bytes"]))).convert("RGB")
    else:
        try:
            return img.convert("RGB")
        except Exception:
            return Image.new("RGB", (512, 512))


# CLIP Feature Extraction (NO VLM generation needed)
def extract_clip_features(clip_model, clip_tokenizer, clip_preprocess,
                          dataset, device, desc="Extracting", max_samples=None):
    """
    Extract CLIP-based features for each sample.
    Only uses the image and the ground-truth text — no VLM generation.
    
    This implements the core FB-M⁴I idea: measuring image-text alignment
    in a pre-trained cross-modal embedding space.
    """
    records = []

    for i, example in enumerate(tqdm(dataset, desc=desc, total=max_samples)):
        if max_samples and i >= max_samples:
            break

        try:
            doc_id = example["id"]
            text = example["text"]
            label = example.get("is_member", None)
            image = extract_image(example)

            # Parse text
            parts = text.split("\n", 1)
            question = parts[0]
            answer = parts[1] if len(parts) > 1 else text

            # Truncate to CLIP context window
            answer_clip = answer[:300]
            question_clip = question[:300]

            # CLIP image embedding
            image_input = clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = clip_model.encode_image(image_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # CLIP text embeddings
            answer_tokens = clip_tokenizer([answer_clip]).to(device)
            question_tokens = clip_tokenizer([question_clip]).to(device)

            with torch.no_grad():
                text_feat = clip_model.encode_text(answer_tokens)
                text_norm = text_feat.norm(dim=-1).item()  # raw norm as complexity proxy
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

                q_feat = clip_model.encode_text(question_tokens)
                q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

            # Cosine similarities
            clip_img_text_sim = float((img_feat @ text_feat.T).squeeze())
            clip_img_question_sim = float((img_feat @ q_feat.T).squeeze())

            records.append({
                "id": doc_id,
                "is_member": label,
                "clip_img_text_sim": clip_img_text_sim,
                "clip_img_question_sim": clip_img_question_sim,
                "clip_text_complexity": text_norm,
            })

        except Exception as e:
            print(f"  ⚠ Error on {example.get('id', i)}: {e}")
            continue

    return pd.DataFrame(records)


# Load Milestone 5 Features
def load_m5_features(split_name):
    """Load pre-computed milestone5 loss-based features."""
    filename_map = {
        "train": "train_features_improved.json",
        "val": "val_features_improved.json",
        "test": "test_features_improved.json",
    }
    path = os.path.join(M5_FEATURE_DIR, filename_map[split_name])
    if not os.path.exists(path):
        print(f"  ⚠ M5 features not found: {path}")
        return None
    
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} M5 features from {os.path.basename(path)}")
    return df


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="M⁴I CLIP-Only Fast Submission"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device: cuda, mps, or cpu (auto-detected if omitted)"
    )
    args = parser.parse_args()

    max_samples = 50 if args.debug else None
    suffix = "_debug" if args.debug else ""

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load CLIP
    print("\nLoading CLIP (ViT-B/32)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    # Load dataset (streaming)
    print("\nLoading dataset...")
    ds_train = load_dataset(DATASET_ID, split="train", streaming=True)
    ds_val = load_dataset(DATASET_ID, split="validation", streaming=True)
    ds_test = load_dataset(DATASET_ID, split="test", streaming=True)

    # Extract CLIP features with caching
    os.makedirs(CACHE_DIR, exist_ok=True)
    splits = {"train": ds_train, "val": ds_val, "test": ds_test}
    clip_dfs = {}

    for split_name, ds in splits.items():
        cache_path = os.path.join(CACHE_DIR, f"{split_name}_clip_only{suffix}.json")

        if os.path.exists(cache_path) and not args.recompute:
            print(f"\nLoading cached {split_name} CLIP features")
            clip_dfs[split_name] = pd.read_json(cache_path)
        else:
            print(f"\n=== Extracting {split_name} CLIP features ===")
            df = extract_clip_features(
                clip_model, clip_tokenizer, clip_preprocess,
                ds, device, desc=split_name.capitalize(),
                max_samples=max_samples,
            )
            df.to_json(cache_path, orient="records", indent=2)
            print(f"  → Cached to {cache_path}")
            clip_dfs[split_name] = df

    # Free CLIP memory
    del clip_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Merge with milestone5 features
    print("\n=== Merging with Milestone 5 features ===")
    merged_dfs = {}
    for split_name in ["train", "val", "test"]:
        clip_df = clip_dfs[split_name]
        m5_df = load_m5_features(split_name)

        if m5_df is not None:
            # Merge on ID
            m5_cols = ["id"] + [c for c in M5_FEATURE_COLS if c in m5_df.columns]
            merged = clip_df.merge(m5_df[m5_cols], on="id", how="inner")
            print(f"  {split_name}: {len(clip_df)} CLIP → {len(merged)} merged")
            merged_dfs[split_name] = merged
        else:
            print(f"  {split_name}: Using CLIP features only")
            merged_dfs[split_name] = clip_df

    df_train = merged_dfs["train"]
    df_val = merged_dfs["val"]
    df_test = merged_dfs["test"]

    # Determine available features
    available_feats = [c for c in ALL_FEATURE_COLS if c in df_train.columns]
    print(f"\n  Using {len(available_feats)} features: {available_feats}")

    # Train classifier
    print("\n" + "=" * 60)
    print(" Training Meta-Classifier")
    print("=" * 60)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[available_feats].values)
    y_train = df_train["is_member"].values

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)

    # Evaluate on validation
    print("\n" + "=" * 60)
    print(" Validation Results — M⁴I CLIP + Loss Features")
    print("=" * 60)

    X_val = scaler.transform(df_val[available_feats].values)
    y_val = df_val["is_member"].values
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, val_probs)
    val_tpr = tpr_at_fpr(y_val, val_probs)
    print(f"  AUC:          {val_auc:.4f}")
    print(f"  TPR@FPR=0.1:  {val_tpr:.4f}")

    print("\n  Feature Importances:")
    importances = sorted(
        zip(available_feats, clf.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances:
        print(f"    {name:25s}: {imp:.4f}")

    # Per-feature AUC
    print("\n  Per-feature AUC (univariate):")
    for col in available_feats:
        try:
            auc_pos = roc_auc_score(y_val, df_val[col].values)
            auc_neg = roc_auc_score(y_val, -df_val[col].values)
            best_auc = max(auc_pos, auc_neg)
            direction = "+" if auc_pos >= auc_neg else "-"
            print(f"    {col:25s}: {best_auc:.4f} ({direction})")
        except ValueError:
            print(f"    {col:25s}: N/A")

    # Generate submission
    print("\n=== Generating Kaggle Submission ===")

    X_test = scaler.transform(df_test[available_feats].values)
    test_probs = clf.predict_proba(X_test)[:, 1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub_path = os.path.join(OUTPUT_DIR, "m4i_clip_attack_submission.csv")
    submission = pd.DataFrame({"id": df_test["id"], "is_member": test_probs})
    submission.to_csv(sub_path, index=False)
    print(f"  Saved to {sub_path}")
    print(f"  Rows: {len(submission)}")
    print(f"  Score stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
