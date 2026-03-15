import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
import os


def tpr_at_fpr(y_true, y_score, target_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    return tpr[idx[-1]] if len(idx) > 0 else 0.0


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    max_length = 1024

    # =========================================================================
    # 1. Load model (only one model needed!)
    # =========================================================================
    model_id = "UBC-SLIME/colx_531_smollm2-135m"
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    # =========================================================================
    # 2. Load datasets
    # =========================================================================
    print("Loading datasets...")
    train_dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="train")
    val_dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="validation")
    test_dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="test")

    # =========================================================================
    # 3. Extract features: original loss + lowercase loss
    # =========================================================================
    def extract_casing_features(dataset, desc="Scoring"):
        records = []
        with torch.no_grad():
            for item in tqdm(dataset, desc=desc):
                text = item["text"]
                doc_id = item.get("id", None)
                label = item.get("is_member", None)

                # Original text loss
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_length
                ).to(device)
                loss_orig = model(**inputs, labels=inputs["input_ids"]).loss.item()

                # Lowercase text loss
                text_lower = text.lower()
                inputs_lower = tokenizer(
                    text_lower, return_tensors="pt", truncation=True, max_length=max_length
                ).to(device)
                loss_lower = model(**inputs_lower, labels=inputs_lower["input_ids"]).loss.item()

                records.append({
                    "id": doc_id,
                    "is_member": label,
                    "loss_orig": loss_orig,
                    "loss_lower": loss_lower,
                    "loss_diff": loss_lower - loss_orig,      # how much worse on lowercased
                    "loss_ratio": loss_lower / (loss_orig + 1e-8),  # ratio
                })
        return pd.DataFrame(records)

    print("\nExtracting features...")
    df_train = extract_casing_features(train_dataset, "Train")
    df_val = extract_casing_features(val_dataset, "Val")
    df_test = extract_casing_features(test_dataset, "Test")

    feature_cols = ["loss_orig", "loss_lower", "loss_diff", "loss_ratio"]

    # =========================================================================
    # 4. Train classifier
    # =========================================================================
    print("\n=== Training Classifier ===")

    X_train = df_train[feature_cols].values
    y_train = df_train["is_member"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["is_member"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    # =========================================================================
    # 5. Evaluate on validation
    # =========================================================================
    print("\n=== Validation Results ===")
    val_probs = clf.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    val_tpr = tpr_at_fpr(y_val, val_probs)
    print(f"Validation AUC:         {val_auc:.4f}")
    print(f"Validation TPR@FPR=0.1: {val_tpr:.4f}")
    print(f"Validation Accuracy:    {accuracy_score(y_val, clf.predict(X_val_scaled)):.4f}")
    print("\nFeature Importances:")
    for name, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:15s}: {imp:.4f}")

    # =========================================================================
    # 6. Predict test set
    # =========================================================================
    X_test = df_test[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]

    os.makedirs("milestone2", exist_ok=True)
    output_file = "milestone2/casing_attack_submission.csv"
    submission = pd.DataFrame({"id": df_test["id"], "is_member": test_probs})
    submission.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    print(f"Score stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")


if __name__ == "__main__":
    main()
