"""
Improved Neighborhood Attack — Dual-Model Membership Inference

Key improvements over the original:
  1. Uses BOTH finetuned models (135M, 360M) + their base models
  2. Replaces slow BERT mask-fill with fast token-drop perturbation
  3. Extracts a rich feature set per sample:
     - finetuned loss (both models)
     - base model loss (both models)
     - reference ratio = base_loss - finetuned_loss (both models)
     - neighborhood gap = mean_neighbor_loss - original_loss (both models)
     - cross-model agreement features
  4. Trains a GradientBoosting classifier on train/val labels

Usage:
    python neighborhood_attack.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_NEIGHBORS = 5         # neighbors per sample (token-drop is fast, so 5 is fine)
DROP_RATIO = 0.10         # fraction of tokens to drop per neighbor
MAX_LENGTH = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tpr_at_fpr(y_true, y_score, target_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    return tpr[idx[-1]] if len(idx) > 0 else 0.0


def compute_loss(model, tokenizer, text, device, max_length=MAX_LENGTH):
    """Average cross-entropy loss for a single text."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


def generate_token_drop_neighbors(text, n_neighbors=NUM_NEIGHBORS,
                                   drop_ratio=DROP_RATIO):
    """
    Fast neighbor generation: randomly drop a fraction of words.
    Much faster than BERT mask-fill, and effective at perturbing the
    surface form while preserving rough semantics.
    """
    words = text.split()
    if len(words) < 5:
        return [text] * n_neighbors

    n_drop = max(1, int(len(words) * drop_ratio))
    neighbors = []
    for _ in range(n_neighbors):
        indices_to_drop = set(random.sample(range(len(words)), n_drop))
        kept = [w for i, w in enumerate(words) if i not in indices_to_drop]
        neighbors.append(" ".join(kept))
    return neighbors


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(models_dict, tokenizer, dataset, device, desc="Scoring"):
    """
    Extract a full feature vector per sample using all available models.

    models_dict: {
        "ft1": finetuned_135m,  "base1": base_135m,
        "ft2": finetuned_360m,  "base2": base_360m,
    }
    """
    ft1 = models_dict["ft1"]
    base1 = models_dict["base1"]
    ft2 = models_dict["ft2"]
    base2 = models_dict["base2"]

    records = []
    for example in tqdm(dataset, desc=desc):
        doc_id = example["id"]
        text = example["text"]
        label = example.get("is_member", None)

        # ---- Original text losses from all 4 models ----
        loss_ft1 = compute_loss(ft1, tokenizer, text, device)
        loss_ft2 = compute_loss(ft2, tokenizer, text, device)
        loss_base1 = compute_loss(base1, tokenizer, text, device)
        loss_base2 = compute_loss(base2, tokenizer, text, device)

        # ---- Reference ratios (higher = more likely member) ----
        ref_diff_1 = loss_base1 - loss_ft1
        ref_diff_2 = loss_base2 - loss_ft2
        ref_ratio_1 = loss_ft1 / (loss_base1 + 1e-8)
        ref_ratio_2 = loss_ft2 / (loss_base2 + 1e-8)

        # ---- Neighborhood losses (only on finetuned models) ----
        neighbors = generate_token_drop_neighbors(text, NUM_NEIGHBORS, DROP_RATIO)

        nb_losses_ft1 = [compute_loss(ft1, tokenizer, nb, device) for nb in neighbors]
        nb_losses_ft2 = [compute_loss(ft2, tokenizer, nb, device) for nb in neighbors]

        mean_nb_ft1 = np.mean(nb_losses_ft1)
        mean_nb_ft2 = np.mean(nb_losses_ft2)

        # Neighborhood gaps (higher = model "memorised" exact wording)
        nb_gap_ft1 = mean_nb_ft1 - loss_ft1
        nb_gap_ft2 = mean_nb_ft2 - loss_ft2

        # Std of neighbor losses (lower variance = more stable = member?)
        nb_std_ft1 = np.std(nb_losses_ft1)
        nb_std_ft2 = np.std(nb_losses_ft2)

        # ---- Cross-model features ----
        loss_diff_ft = loss_ft1 - loss_ft2  # agreement between two finetuned
        gap_diff = nb_gap_ft1 - nb_gap_ft2  # do both models show same pattern?

        # ---- Text-level features ----
        words = text.split()
        text_len = len(words)
        unique_ratio = len(set(words)) / text_len if text_len > 0 else 0

        records.append({
            "id": doc_id,
            "is_member": label,
            # Finetuned model losses
            "loss_ft1": loss_ft1,
            "loss_ft2": loss_ft2,
            # Base model losses
            "loss_base1": loss_base1,
            "loss_base2": loss_base2,
            # Reference ratios
            "ref_diff_1": ref_diff_1,
            "ref_diff_2": ref_diff_2,
            "ref_ratio_1": ref_ratio_1,
            "ref_ratio_2": ref_ratio_2,
            # Neighborhood gaps
            "nb_gap_ft1": nb_gap_ft1,
            "nb_gap_ft2": nb_gap_ft2,
            "nb_std_ft1": nb_std_ft1,
            "nb_std_ft2": nb_std_ft2,
            # Cross-model
            "loss_diff_ft": loss_diff_ft,
            "gap_diff": gap_diff,
            # Text features
            "text_len": text_len,
            "unique_ratio": unique_ratio,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "loss_ft1", "loss_ft2",
    "loss_base1", "loss_base2",
    "ref_diff_1", "ref_diff_2",
    "ref_ratio_1", "ref_ratio_2",
    "nb_gap_ft1", "nb_gap_ft2",
    "nb_std_ft1", "nb_std_ft2",
    "loss_diff_ft", "gap_diff",
    "text_len", "unique_ratio",
]


def main():
    # ---- Device -----------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # ---- Load tokenizer (shared by 135M family) ---------------------------
    tokenizer = AutoTokenizer.from_pretrained("UBC-SLIME/colx_531_smollm2-135m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load ALL 4 models ------------------------------------------------
    print("Loading finetuned model 1 (135M)...")
    ft1 = AutoModelForCausalLM.from_pretrained(
        "UBC-SLIME/colx_531_smollm2-135m"
    ).to(device).eval()

    print("Loading base model 1 (135M)...")
    base1 = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M"
    ).to(device).eval()

    print("Loading finetuned model 2 (360M)...")
    ft2 = AutoModelForCausalLM.from_pretrained(
        "UBC-SLIME/colx_531_smollm2-360m"
    ).to(device).eval()

    print("Loading base model 2 (360M)...")
    base2 = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M"
    ).to(device).eval()

    models = {"ft1": ft1, "base1": base1, "ft2": ft2, "base2": base2}

    # ---- Load datasets ----------------------------------------------------
    print("\nLoading datasets...")
    train_dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="train")
    val_dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="validation")
    test_dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="test")

    # ---- Extract features -------------------------------------------------
    print("\n=== Extracting features ===")
    df_train = extract_features(models, tokenizer, train_dataset, device, "Train")
    df_val = extract_features(models, tokenizer, val_dataset, device, "Val")
    df_test = extract_features(models, tokenizer, test_dataset, device, "Test")

    # ---- Train classifier -------------------------------------------------
    print("\n=== Training GradientBoosting Classifier ===")

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["is_member"].values
    X_val = df_val[FEATURE_COLS].values
    y_val = df_val["is_member"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_train_s, y_train)

    # ---- Evaluate on validation -------------------------------------------
    print("\n" + "=" * 60)
    print(" Validation Results — Improved Neighborhood Attack")
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
        print(f"    {name:20s}: {imp:.4f}")

    # ---- Predict on test --------------------------------------------------
    print("\n=== Generating test submission ===")

    X_test = df_test[FEATURE_COLS].values
    X_test_s = scaler.transform(X_test)
    test_probs = clf.predict_proba(X_test_s)[:, 1]

    os.makedirs("milestone2", exist_ok=True)
    output_file = "milestone2/neighborhood_attack_submission.csv"
    submission = pd.DataFrame({"id": df_test["id"], "score": test_probs})
    submission.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    print(f"Score stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    # Save feature details
    detail_file = "milestone2/neighborhood_attack_features.csv"
    df_test[["id"] + FEATURE_COLS].to_csv(detail_file, index=False)
    print(f"Feature details saved to {detail_file}")


if __name__ == "__main__":
    main()
