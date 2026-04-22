"""
Milestone 5: Lightweight Single-Model MIA via Complexity Calibration
Pipeline: VLM Feature Extraction -> XGBoost Classifier 

Gemini was used to help re-compile and optimize the script
"""

import os
import io
import zlib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from transformers import SmolVLMForConditionalGeneration, AutoProcessor

# =============================================================================
# 1. Configuration
# =============================================================================
MIN_K_RATIO = 0.20
BATCH_SIZE = 4 

# Subset Sizes
TRAIN_SIZE = 500
VAL_SIZE = 100
TEST_SIZE = 500

FINETUNED_ID = "UBC-SLIME/colx_585_vlm"
DATASET_ID = "UBC-SLIME/colx585_group_project_data"

FEATURE_COLS = [
    "loss_ft_avg", "perplexity_ft", "min_k_loss_ft", 
    "max_token_loss", "std_token_loss", "zlib_ratio_ft", "text_len"
]

# =============================================================================
# 2. VLM Feature Extraction
# =============================================================================
def extract_image(example):
    img = example["image"]
    if hasattr(img, "convert"): 
        return img.convert("RGB").resize((364, 364), Image.BICUBIC)
    elif isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(bytes(img["bytes"]))).convert("RGB").resize((364, 364), Image.BICUBIC)
    return Image.new("RGB", (364, 364)).resize((364, 364), Image.BICUBIC)

def get_losses(model, processor, images, texts, device, dtype):
    prompts, ans_texts = [], []
    for t in texts:
        # Safe truncation to prevent OOM memory spikes
        ans = " ".join(t.split("\n", 1)[-1].split()[:300]) 
        ans_texts.append(ans)
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]},
               {"role": "assistant", "content": [{"type": "text", "text": ans}]}]
        prompts.append(processor.apply_chat_template(msg, tokenize=False))
    
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device)
    inputs = {k: v.to(dtype) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for i in range(len(texts)):
        ans_len = len(processor.tokenizer(ans_texts[i] + "<|im_end|>", add_special_tokens=False).input_ids)
        labels[i, : (inputs["attention_mask"][i].sum().item() - ans_len)] = -100 
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(logits[..., :-1, :].contiguous().view(-1, logits.size(-1)), 
                             labels[..., 1:].contiguous().view(-1)).view(len(texts), -1)
    return [token_losses[i][labels[i, 1:] != -100].cpu().float().numpy() for i in range(len(texts))]

def run_extraction(ft_model, processor, dataset, device, dtype, name):
    records = []
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=f"Extracting {name}"):
        batch = [dataset[j] for j in range(i, min(i + BATCH_SIZE, len(dataset)))]
        imgs, texts = [extract_image(ex) for ex in batch], [ex["text"] for ex in batch]
        
        try:
            ft_l_batch = get_losses(ft_model, processor, imgs, texts, device, dtype)
            
            for j, ex in enumerate(batch):
                ft_l = ft_l_batch[j]
                if len(ft_l) == 0: continue
                
                k = max(1, int(len(ft_l) * MIN_K_RATIO))
                zlib_bytes = len(zlib.compress(bytes(ex["text"], 'utf-8')))
                loss_avg = float(np.mean(ft_l))
                
                records.append({
                    "id": ex["id"], 
                    "is_member": ex.get("is_member", 0),
                    "loss_ft_avg": loss_avg,
                    "perplexity_ft": float(np.exp(loss_avg)),
                    "min_k_loss_ft": float(np.mean(np.sort(ft_l)[-k:])),
                    "max_token_loss": float(np.max(ft_l)),
                    "std_token_loss": float(np.std(ft_l)),
                    "zlib_ratio_ft": float(loss_avg / zlib_bytes) if zlib_bytes > 0 else 0,
                    "text_len": len(ft_l)
                })
        except Exception as e:
            print(f"\nError on batch {i}: {e}")
            continue
    return pd.DataFrame(records)

def tpr_at_fpr(y_true, y_score, target_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    return tpr[idx[-1]] if len(idx) > 0 else 0.0

# =============================================================================
# 3. Main Execution
# =============================================================================
def main():
    # Phase 1: Feature Extraction (Auto-Skips if CSVs exist)
    if os.path.exists("train_features.csv") and os.path.exists("val_features.csv") and os.path.exists("test_features.csv"):
        print("Found existing feature CSVs. Skipping VLM Extraction...")
        df_train = pd.read_csv("train_features.csv")
        df_val = pd.read_csv("val_features.csv")
        df_test = pd.read_csv("test_features.csv")
    else:
        print("Feature CSVs not found. Starting VLM extraction...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Apple Silicon (M-series) check for local execution
        if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Running on: {device} | Dtype: {dtype}")
        
        print("Loading Finetuned Model and Processor...")
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        ft = SmolVLMForConditionalGeneration.from_pretrained(FINETUNED_ID, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
        
        print("Loading and Shuffling dataset...")
        ds = load_dataset(DATASET_ID)
        
        df_train = run_extraction(ft, processor, ds["train"].shuffle(seed=42).select(range(TRAIN_SIZE)), device, dtype, "Train")
        df_val = run_extraction(ft, processor, ds["validation"].shuffle(seed=42).select(range(VAL_SIZE)), device, dtype, "Val")
        df_test = run_extraction(ft, processor, ds["test"].shuffle(seed=42).select(range(TEST_SIZE)), device, dtype, "Test")
        
        print("\nSaving features to CSV...")
        df_train.to_csv("train_features.csv", index=False)
        df_val.to_csv("val_features.csv", index=False)
        df_test.to_csv("test_features.csv", index=False)
        
        # Free memory before training
        del ft, processor
        if device == "cuda":
            torch.cuda.empty_cache()

    # Phase 2: XGBoost Training & Evaluation
    print("\n" + "="*55)
    print("=== Training XGBoost Classifier ===")
    print("="*55)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS])
    y_train = df_train["is_member"]
    
    clf = XGBClassifier(
        n_estimators=150, 
        max_depth=3, 
        learning_rate=0.05, 
        subsample=0.8, 
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    X_val = scaler.transform(df_val[FEATURE_COLS])
    y_val = df_val["is_member"]
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    print(f"Validation AUC:         {roc_auc_score(y_val, val_probs):.4f}")
    print(f"Validation TPR@FPR=0.1: {tpr_at_fpr(y_val, val_probs):.4f}")
    
    print("\nFeature Importances:")
    for name, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:25s}: {imp:.4f}")
    
    # Phase 3: Kaggle Submission
    X_test = scaler.transform(df_test[FEATURE_COLS])
    test_probs = clf.predict_proba(X_test)[:, 1]
    sub_path = "fast_submission.csv"
    pd.DataFrame({"id": df_test["id"], "score": test_probs}).to_csv(sub_path, index=False)
    print(f"\nDone! Saved Kaggle submission to: {sub_path}")

if __name__ == "__main__": 
    main()