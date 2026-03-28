import torch
import transformers
import datasets
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

print(torch.__version__)
print(transformers.__version__)
print(datasets.__version__)
# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model_id = "UBC-SLIME/colx_585_vlm"

# Load processor
processor = AutoProcessor.from_pretrained(model_id)
processor.image_processor.do_image_splitting = False
processor.image_processor.size = {"longest_edge": 512}
processor.image_processor.max_image_size = {"longest_edge": 512}

# Load model

model = SmolVLMForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    _attn_implementation="sdpa",
    trust_remote_code=True
).to(device)
model.eval() # Set to evaluation mode
print("Model loaded successfully.")

# Load dataset
dataset_id = "UBC-SLIME/colx585_group_project_data"

def compute_loss(example, processor, model, device):
    # Extract image
    if isinstance(example['image'], dict) and 'bytes' in example['image']:
        try:
            image = Image.open(io.BytesIO(example['image']['bytes'])).convert("RGB")
        except Exception:
            # Fallback if image data is raw bytes
            import base64
            image = Image.open(io.BytesIO(base64.b64decode(example['image']['bytes']))).convert("RGB")
    else:
        try:
            image = example['image'].convert("RGB")
        except Exception:
            image = Image.new('RGB', (512, 512)) # dummy if fails
    
    parts = example['text'].split('\n', 1)
    user_text = parts[0]
    assistant_text = parts[1] if len(parts) > 1 else ""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    if device == "cuda":
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run on a small subset of data")
    args = parser.parse_args()

    max_samples = 100 if args.debug else None

    # Process and collect features for training LR
    train_features_path = "train_features.json"
    if os.path.exists(train_features_path) and not args.debug:
        print(f"Loading cached train features from {train_features_path}")
        with open(train_features_path, 'r') as f:
            train_data = json.load(f)
    else:
        print("Computing features for train split...")
        ds_train = load_dataset(dataset_id, split="train", streaming=True)
        train_data = []
        pbar = tqdm(total=max_samples if max_samples else 6000, desc="Train split")
        for i, example in enumerate(ds_train):
            if max_samples and i >= max_samples:
                break
            try:
                loss = compute_loss(example, processor, model, device)
                train_data.append({
                    "id": example["id"],
                    "loss": loss,
                    "is_member": example.get("is_member", 1)  # Default to 1 if it's train just in case
                })
            except Exception as e:
                print(f"Error processing train sample {example.get('id', i)}: {e}")
            pbar.update(1)
        pbar.close()
        if not args.debug:
            with open(train_features_path, 'w') as f:
                json.dump(train_data, f)
    
    # Train Logistic Regression
    print("Training Logistic Regression")
    df_train = pd.DataFrame(train_data)
    if df_train.empty:
        print("No training data found. Exiting.")
        return
        
    X_train = df_train[['loss']]
    y_train = df_train['is_member']
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train, y_train)

    # Process validation split and evaluate
    val_features_path = "val_features.json"
    if os.path.exists(val_features_path) and not args.debug:
        print(f"Loading cached validation features from {val_features_path}")
        with open(val_features_path, 'r') as f:
            val_data = json.load(f)
    else:
        print("Computing features for validation split...")
        ds_val = load_dataset(dataset_id, split="validation", streaming=True)
        val_data = []
        pbar = tqdm(total=max_samples if max_samples else 1200, desc="Validation split")
        for i, example in enumerate(ds_val):
            if max_samples and i >= max_samples:
                break
            try:
                loss = compute_loss(example, processor, model, device)
                val_data.append({
                    "id": example["id"],
                    "loss": loss,
                    "is_member": example.get("is_member", 0)  # Default non-member for val/test if not present just in case
                })
            except Exception as e:
                print(f"Error processing validation sample {example.get('id', i)}: {e}")
            pbar.update(1)
        pbar.close()
        if not args.debug:
            with open(val_features_path, 'w') as f:
                json.dump(val_data, f)
                
    df_val = pd.DataFrame(val_data)
    if not df_val.empty and 'is_member' in df_val.columns:
        X_val = df_val[['loss']]
        y_val = df_val['is_member']
        
        # We need probabilities for roc_auc_score
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        try:
            auc_score = roc_auc_score(y_val, y_val_prob)
            fpr, tpr, _ = roc_curve(y_val, y_val_prob)
            tpr_at_fpr_10 = tpr[fpr <= 0.1][-1] if len(tpr[fpr <= 0.1]) > 0 else 0.0
            print(f"Validation ROC AUC Score: {auc_score:.4f}")
            print(f"Validation TPR@FPR=0.1: {tpr_at_fpr_10:.4f}")
        except ValueError as e:
            print(f"Could not calculate metrics: {e}")

    # Process test split and predict
    test_features_path = "test_features.json"
    if os.path.exists(test_features_path) and not args.debug:
        print(f"Loading cached test features from {test_features_path}")
        with open(test_features_path, 'r') as f:
            test_data = json.load(f)
    else:
        print("Computing features for test split...")
        ds_test = load_dataset(dataset_id, split="test", streaming=True)
        test_data = []
        pbar = tqdm(total=max_samples if max_samples else 6000, desc="Test split")
        for i, example in enumerate(ds_test):
            if max_samples and i >= max_samples:
                break
            try:
                loss = compute_loss(example, processor, model, device)
                test_data.append({
                    "id": example["id"],
                    "loss": loss
                })
            except Exception as e:
                print(f"Error processing test sample {example.get('id', i)}: {e}")
            pbar.update(1)
        pbar.close()
        if not args.debug:
            with open(test_features_path, 'w') as f:
                json.dump(test_data, f)

    # Predict
    print("Generating predictions...")
    df_test = pd.DataFrame(test_data)
    if df_test.empty:
        print("No test data found. Exiting.")
        return
        
    X_test = df_test[['loss']]
    # Assign higher continuous scores (probabilities) to predicted members
    predictions_prob = clf.predict_proba(X_test)[:, 1]
    df_test['is_member'] = predictions_prob

    # Export to CSV
    output_dir = r"milestone4\output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, "baseline_submission.csv")
    df_test[['id', 'is_member']].to_csv(output_filename, index=False)
    print(f"Done! Predictions saved to {output_filename}")

if __name__ == "__main__":
    main()
