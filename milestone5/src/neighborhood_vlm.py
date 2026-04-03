"""
VLM Neighborhood Attack — Dual-Model Membership Inference for Vision Language Models

Key design (adapted from milestone3 text-only neighborhood attack):
  1. Uses BOTH finetuned VLM + base (pre-finetune) VLM
  2. Text perturbation via token-drop (image fixed) → text memorization signal
  3. Image perturbation via patch masking / noise / crop (text fixed) → image memorization signal
  4. Cross-modal binding features → detects if exact (image, text) pair was seen
  5. XGBoost classifier (falls back to GradientBoosting if xgboost not installed)

Evaluation metrics: AUC and TPR@FPR=0.1

Usage:
    python neighborhood_vlm.py              # full run
    python neighborhood_vlm.py --debug      # small subset for testing
    python neighborhood_vlm.py --recompute  # re-extract features (ignore cache)
"""

import os
import io
import random
import json
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

# Try XGBoost first; fall back to sklearn GBC
try:
    from xgboost import XGBClassifier

    USE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier

    USE_XGBOOST = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_TEXT_NEIGHBORS = 5  # number of text-perturbed neighbors per sample
NUM_IMAGE_NEIGHBORS = 5  # number of image-perturbed neighbors per sample
TEXT_DROP_RATIO = 0.10  # fraction of words to drop in text perturbation
IMAGE_NOISE_STD = 25.0  # std-dev for Gaussian noise (pixel range 0-255)
PATCH_MASK_RATIO = 0.15  # fraction of patches to mask
PATCH_SIZE = 32  # pixel size of each patch for masking

FINETUNED_MODEL_ID = "UBC-SLIME/colx_585_vlm"
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASET_ID = "UBC-SLIME/colx585_group_project_data"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "features_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

FEATURE_COLS = [
    # --- Base losses (4) ---
    "loss_ft",
    "loss_base",
    "ref_diff",
    "ref_ratio",
    # --- Text neighborhood (4) ---
    "text_nb_gap_ft",
    "text_nb_gap_base",
    "text_nb_std_ft",
    "text_nb_relative_gap",
    # --- Image neighborhood (4) ---
    "img_nb_gap_ft",
    "img_nb_gap_base",
    "img_nb_std_ft",
    "img_nb_relative_gap",
    # --- Cross-modal binding (3) ---
    "cross_modal_gap",
    "binding_score",
    "total_nb_gap",
    # --- Text statistics (2) ---
    "text_len",
    "unique_ratio",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tpr_at_fpr(y_true, y_score, target_fpr=0.1):
    """Compute TPR at a given FPR threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    return tpr[idx[-1]] if len(idx) > 0 else 0.0


def extract_image(example):
    """Extract a PIL Image from a dataset example (handles multiple formats)."""
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


def compute_vlm_loss(model, processor, image, text, device):
    """
    Compute average cross-entropy loss for an (image, text) pair.

    The text is formatted as:
        <user question>\n<assistant answer>
    using the VLM chat template.
    """
    parts = text.split("\n", 1)
    user_text = parts[0]
    assistant_text = parts[1] if len(parts) > 1 else ""

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    # Cast float32 → bfloat16 on CUDA for speed
    if device == "cuda":
        inputs = {
            k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
            for k, v in inputs.items()
        }

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    return outputs.loss.item()


# ---------------------------------------------------------------------------
# Text Perturbation
# ---------------------------------------------------------------------------


def generate_text_neighbors(
    text, n_neighbors=NUM_TEXT_NEIGHBORS, drop_ratio=TEXT_DROP_RATIO
):
    """
    Generate text neighbors by randomly dropping words from the assistant response.

    Only the *answer* part (after the first newline) is perturbed,
    while the user question stays intact.  This is because the model's
    memorization signal lives in the specific answer to a specific question.
    """
    parts = text.split("\n", 1)
    user_text = parts[0]
    assistant_text = parts[1] if len(parts) > 1 else ""

    words = assistant_text.split()
    if len(words) < 5:
        return [text] * n_neighbors

    n_drop = max(1, int(len(words) * drop_ratio))
    neighbors = []
    for _ in range(n_neighbors):
        indices_to_drop = set(random.sample(range(len(words)), n_drop))
        kept = [w for i, w in enumerate(words) if i not in indices_to_drop]
        perturbed_assistant = " ".join(kept)
        neighbors.append(user_text + "\n" + perturbed_assistant)
    return neighbors


# ---------------------------------------------------------------------------
# Image Perturbation  (VLM-specific)
# ---------------------------------------------------------------------------


def add_gaussian_noise(image, std=IMAGE_NOISE_STD):
    """Add Gaussian noise to pixel values."""
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def mask_random_patches(image, mask_ratio=PATCH_MASK_RATIO, patch_size=PATCH_SIZE):
    """
    Divide image into non-overlapping patches and mask a random subset
    with the image's mean color.  Inspired by MAE (Masked Autoencoders).
    """
    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    n_patches_h = max(1, h // patch_size)
    n_patches_w = max(1, w // patch_size)
    total_patches = n_patches_h * n_patches_w

    if total_patches < 2:
        return image

    n_mask = max(1, int(total_patches * mask_ratio))
    patch_indices = random.sample(range(total_patches), min(n_mask, total_patches))
    mean_color = arr.mean(axis=(0, 1)).astype(np.uint8)

    for idx in patch_indices:
        pi = idx // n_patches_w
        pj = idx % n_patches_w
        y0, x0 = pi * patch_size, pj * patch_size
        arr[y0 : y0 + patch_size, x0 : x0 + patch_size] = mean_color

    return Image.fromarray(arr)


def random_crop_resize(image, crop_ratio_range=(0.70, 0.90)):
    """Randomly crop 70-90 % of the image and resize back to original dimensions."""
    w, h = image.size
    ratio = random.uniform(*crop_ratio_range)
    new_w, new_h = int(w * ratio), int(h * ratio)

    left = random.randint(0, max(0, w - new_w))
    top = random.randint(0, max(0, h - new_h))

    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


def generate_image_neighbors(image, n_neighbors=NUM_IMAGE_NEIGHBORS):
    """
    Generate perturbed image neighbors by cycling through three strategies:
      1. Gaussian noise
      2. Patch masking  (MAE-style)
      3. Random crop + resize
    """
    strategies = [add_gaussian_noise, mask_random_patches, random_crop_resize]
    neighbors = []
    for i in range(n_neighbors):
        strategy = strategies[i % len(strategies)]
        neighbors.append(strategy(image))
    return neighbors


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_features(
    ft_model, base_model, processor, dataset, device, desc="Scoring", max_samples=None
):
    """
    Extract a 17-dimensional feature vector per sample using both models.

    For each sample we compute:
      - original losses from finetuned & base model
      - text-neighborhood losses   (token-drop text, keep image)
      - image-neighborhood losses  (perturb image, keep text)
      - cross-modal binding scores
      - text statistics
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

            # ---- Original (image, text) losses from both models ----
            loss_ft = compute_vlm_loss(ft_model, processor, image, text, device)
            loss_base = compute_vlm_loss(base_model, processor, image, text, device)

            ref_diff = loss_base - loss_ft
            ref_ratio = loss_ft / (loss_base + 1e-8)

            # ---- Text neighborhood: perturb text, keep image fixed ----
            text_neighbors = generate_text_neighbors(text)
            text_nb_losses_ft = [
                compute_vlm_loss(ft_model, processor, image, nb, device)
                for nb in text_neighbors
            ]
            text_nb_losses_base = [
                compute_vlm_loss(base_model, processor, image, nb, device)
                for nb in text_neighbors
            ]

            text_nb_gap_ft = np.mean(text_nb_losses_ft) - loss_ft
            text_nb_gap_base = np.mean(text_nb_losses_base) - loss_base
            text_nb_std_ft = np.std(text_nb_losses_ft)
            text_nb_relative_gap = text_nb_gap_ft - text_nb_gap_base

            # ---- Image neighborhood: perturb image, keep text fixed ----
            image_neighbors = generate_image_neighbors(image)
            img_nb_losses_ft = [
                compute_vlm_loss(ft_model, processor, nb_img, text, device)
                for nb_img in image_neighbors
            ]
            img_nb_losses_base = [
                compute_vlm_loss(base_model, processor, nb_img, text, device)
                for nb_img in image_neighbors
            ]

            img_nb_gap_ft = np.mean(img_nb_losses_ft) - loss_ft
            img_nb_gap_base = np.mean(img_nb_losses_base) - loss_base
            img_nb_std_ft = np.std(img_nb_losses_ft)
            img_nb_relative_gap = img_nb_gap_ft - img_nb_gap_base

            # ---- Cross-modal binding features ----
            # If the exact (image, text) pair was seen during training,
            # BOTH text and image perturbations should increase the loss.
            cross_modal_gap = text_nb_gap_ft - img_nb_gap_ft
            binding_score = text_nb_gap_ft * img_nb_gap_ft
            total_nb_gap = text_nb_gap_ft + img_nb_gap_ft

            # ---- Text-level statistics ----
            words = text.split()
            text_len = len(words)
            unique_ratio = len(set(words)) / text_len if text_len > 0 else 0

            records.append(
                {
                    "id": doc_id,
                    "is_member": label,
                    # Base losses
                    "loss_ft": loss_ft,
                    "loss_base": loss_base,
                    "ref_diff": ref_diff,
                    "ref_ratio": ref_ratio,
                    # Text neighborhood
                    "text_nb_gap_ft": text_nb_gap_ft,
                    "text_nb_gap_base": text_nb_gap_base,
                    "text_nb_std_ft": text_nb_std_ft,
                    "text_nb_relative_gap": text_nb_relative_gap,
                    # Image neighborhood
                    "img_nb_gap_ft": img_nb_gap_ft,
                    "img_nb_gap_base": img_nb_gap_base,
                    "img_nb_std_ft": img_nb_std_ft,
                    "img_nb_relative_gap": img_nb_relative_gap,
                    # Cross-modal
                    "cross_modal_gap": cross_modal_gap,
                    "binding_score": binding_score,
                    "total_nb_gap": total_nb_gap,
                    # Text stats
                    "text_len": text_len,
                    "unique_ratio": unique_ratio,
                }
            )

        except Exception as e:
            print(f"  ⚠ Error processing sample {example.get('id', i)}: {e}")
            continue

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VLM Neighborhood Attack — Membership Inference"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run on a small subset (50 samples)"
    )
    parser.add_argument(
        "--recompute", action="store_true", help="Re-extract features (ignore cache)"
    )
    args = parser.parse_args()

    max_samples = 50 if args.debug else None

    # ---- Device ----
    if torch.cuda.is_available():
        device = "cuda"
        print("Using device: CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using device: CPU")

    # ---- Load processor (shared between finetuned & base) ----
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}

    # ---- Load models ----
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("Loading finetuned VLM...")
    ft_model = (
        SmolVLMForConditionalGeneration.from_pretrained(
            FINETUNED_MODEL_ID,
            torch_dtype=dtype,
            _attn_implementation="sdpa",
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )

    print("Loading base VLM...")
    base_model = (
        SmolVLMForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
            _attn_implementation="sdpa",
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )

    # ---- Load datasets (streaming to avoid large downloads) ----
    print("\nLoading datasets...")
    ds_train = load_dataset(DATASET_ID, split="train", streaming=True)
    ds_val = load_dataset(DATASET_ID, split="validation", streaming=True)
    ds_test = load_dataset(DATASET_ID, split="test", streaming=True)

    # ---- Feature extraction with caching ----
    os.makedirs(CACHE_DIR, exist_ok=True)
    suffix = "_debug" if args.debug else ""  # separate cache for debug vs full

    # Train
    train_cache = os.path.join(CACHE_DIR, f"train_nb{suffix}.json")
    if os.path.exists(train_cache) and not args.recompute:
        print(f"\nLoading cached train features from {train_cache}")
        df_train = pd.read_json(train_cache)
    else:
        print("\n=== Extracting train features ===")
        df_train = extract_features(
            ft_model, base_model, processor, ds_train, device, "Train", max_samples
        )
        df_train.to_json(train_cache, orient="records", indent=2)
        print(f"  → cached to {train_cache}")

    # Validation
    val_cache = os.path.join(CACHE_DIR, f"val_nb{suffix}.json")
    if os.path.exists(val_cache) and not args.recompute:
        print(f"\nLoading cached val features from {val_cache}")
        df_val = pd.read_json(val_cache)
    else:
        print("\n=== Extracting val features ===")
        df_val = extract_features(
            ft_model, base_model, processor, ds_val, device, "Val", max_samples
        )
        df_val.to_json(val_cache, orient="records", indent=2)
        print(f"  → cached to {val_cache}")

    # Test
    test_cache = os.path.join(CACHE_DIR, f"test_nb{suffix}.json")
    if os.path.exists(test_cache) and not args.recompute:
        print(f"\nLoading cached test features from {test_cache}")
        df_test = pd.read_json(test_cache)
    else:
        print("\n=== Extracting test features ===")
        df_test = extract_features(
            ft_model, base_model, processor, ds_test, device, "Test", max_samples
        )
        df_test.to_json(test_cache, orient="records", indent=2)
        print(f"  → cached to {test_cache}")

    # ---- Train classifier ----
    print("\n=== Training Classifier ===")

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["is_member"].values
    X_val = df_val[FEATURE_COLS].values
    y_val = df_val["is_member"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    if USE_XGBOOST:
        print("  Classifier: XGBoost")
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
    else:
        print("  Classifier: GradientBoosting (install xgboost for better results)")
        clf = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    clf.fit(X_train_s, y_train)

    # ---- Evaluate on validation ----
    print("\n" + "=" * 60)
    print(" Validation Results — VLM Neighborhood Attack")
    print("=" * 60)

    val_probs = clf.predict_proba(X_val_s)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    val_tpr = tpr_at_fpr(y_val, val_probs)
    val_acc = accuracy_score(y_val, clf.predict(X_val_s))

    print(f"  AUC:          {val_auc:.4f}")
    print(f"  TPR@FPR=0.1:  {val_tpr:.4f}")
    print(f"  Accuracy:     {val_acc:.4f}")

    print("\n  Feature Importances (top 10):")
    importances = sorted(
        zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances[:10]:
        print(f"    {name:25s}: {imp:.4f}")

    # ---- Predict on test ----
    print("\n=== Generating test submission ===")

    X_test = df_test[FEATURE_COLS].values
    X_test_s = scaler.transform(X_test)
    test_probs = clf.predict_proba(X_test_s)[:, 1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(OUTPUT_DIR, "neighborhood_vlm_submission.csv")
    submission = pd.DataFrame({"id": df_test["id"], "score": test_probs})
    submission.to_csv(output_file, index=False)
    print(f"  Predictions saved to {output_file}")
    print(f"  Score stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    # Save detailed features for analysis
    detail_file = os.path.join(OUTPUT_DIR, "neighborhood_vlm_features.csv")
    df_test[["id"] + FEATURE_COLS].to_csv(detail_file, index=False)
    print(f"  Feature details saved to {detail_file}")


if __name__ == "__main__":
    main()
