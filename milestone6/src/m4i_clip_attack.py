"""
M⁴I-Adapted Membership Inference Attack for VLMs
=================================================

Adapts two core strategies from:
  Hu et al., "M⁴I: Multi-modal Models Membership Inference" (NeurIPS 2022)

to the Kaggle VLM MIA competition (UBC-SLIME/colx585_group_project_data).

Method 1 — Metric-Based M⁴I (MB-M⁴I):
  Generate text with the finetuned SmolVLM for each image, then compare
  to the ground-truth text using ROUGE-L and token overlap.
  Members have higher similarity because the model memorized them.

Method 2 — Feature-Based M⁴I (FB-M⁴I):
  Use CLIP (ViT-B/32) as a cross-modal feature extractor. Compute cosine
  similarities between image, ground-truth text, and generated text
  embeddings. Members show tighter image-text binding in CLIP space.

Both feature sets are combined into an XGBoost meta-classifier.

Usage:
    python m4i_clip_attack.py               # full run
    python m4i_clip_attack.py --debug       # small subset (~50 per split)
    python m4i_clip_attack.py --recompute   # ignore cached features
    python m4i_clip_attack.py --device cuda # use cuda
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

# Lazy imports for heavy libraries (fail fast with clear messages)
try:
    from rouge_score import rouge_scorer
except ImportError:
    raise ImportError("Install rouge-score: pip install rouge-score")

try:
    import open_clip
except ImportError:
    raise ImportError("Install open_clip: pip install open_clip_torch")

try:
    from xgboost import XGBClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier

    XGBClassifier = None
    print("WARNING: xgboost not installed, falling back to sklearn GBC")

from transformers import AutoProcessor, SmolVLMForConditionalGeneration


# Configuration
FINETUNED_MODEL_ID = "UBC-SLIME/colx_585_vlm"
BASE_PROCESSOR_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASET_ID = "UBC-SLIME/colx585_group_project_data"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

MAX_GEN_TOKENS = 300  # max tokens for SmolVLM generation
GENERATION_PROMPT = "Describe this image."

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4i_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")

FEATURE_COLS = [
    # MB-M⁴I features (metric-based)
    "rouge_l",
    "token_overlap",
    # FB-M⁴I features (feature-based / CLIP)
    "clip_img_text_sim",
    "clip_img_gen_sim",
    "clip_text_gen_sim",
    "clip_sim_gap",
    # Auxiliary
    "gen_length",
]


# Helpers
def tpr_at_fpr(y_true, y_score, target_fpr=0.1):
    """Compute TPR at a given FPR threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    return tpr[idx[-1]] if len(idx) > 0 else 0.0


def extract_image(example):
    """Extract a PIL Image from a HuggingFace dataset example."""
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


# SmolVLM Text Generation
def generate_text(model, processor, image, device):
    """
    Generate text from the finetuned SmolVLM given an image.
    Uses greedy decoding for deterministic, most-memorized output.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": GENERATION_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    # Cast to match model dtype
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(model_dtype) if v.is_floating_point() else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=False,  # greedy
        )

    # Decode only the generated tokens (skip the prompt)
    input_len = inputs["input_ids"].shape[-1]
    generated = processor.tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )
    return generated.strip()


# =============================================================================
# MB-M⁴I: Metric-Based Features
# =============================================================================
def compute_metric_features(ground_truth_text, generated_text, scorer):
    """
    Compute metric-based MIA features by comparing generated text to ground truth.
    
    Adapted from M⁴I's MB attack which used ROUGE scores between
    model-generated captions and reference captions.
    """
    # Parse ground truth: format is "question\nanswer"
    parts = ground_truth_text.split("\n", 1)
    gt_answer = parts[1] if len(parts) > 1 else ground_truth_text

    # ROUGE-L F1
    rouge_scores = scorer.score(gt_answer, generated_text)
    rouge_l = rouge_scores["rougeL"].fmeasure

    # Token overlap ratio (bag-of-words Jaccard-like)
    gt_tokens = set(gt_answer.lower().split())
    gen_tokens = set(generated_text.lower().split())
    if len(gt_tokens) == 0 or len(gen_tokens) == 0:
        token_overlap = 0.0
    else:
        intersection = gt_tokens & gen_tokens
        token_overlap = len(intersection) / len(gt_tokens | gen_tokens)

    return {
        "rouge_l": rouge_l,
        "token_overlap": token_overlap,
    }


# FB-M⁴I: Feature-Based (CLIP) Features
def compute_clip_features(image, ground_truth_text, generated_text,
                          clip_model, clip_tokenizer, clip_preprocess, device):
    """
    Compute feature-based MIA features using CLIP embeddings.
    
    Adapted from M⁴I's FB attack which used a custom Multi-modal Feature
    Extractor (MFE) to project images and text into a shared space and
    measured Euclidean distance. We modernize this with CLIP cosine similarity.
    """
    # Parse ground truth answer
    parts = ground_truth_text.split("\n", 1)
    gt_answer = parts[1] if len(parts) > 1 else ground_truth_text

    # Truncate texts to CLIP's context window (~77 tokens → ~300 chars safe)
    gt_answer_clip = gt_answer[:300]
    gen_text_clip = generated_text[:300]

    # Image embedding
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = clip_model.encode_image(image_input)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    # Text embeddings
    gt_tokens = clip_tokenizer([gt_answer_clip]).to(device)
    gen_tokens = clip_tokenizer([gen_text_clip]).to(device)

    with torch.no_grad():
        gt_features = clip_model.encode_text(gt_tokens)
        gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)

        gen_features = clip_model.encode_text(gen_tokens)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

    # Cosine similarities (already normalized, so dot product = cosine sim)
    clip_img_text_sim = float((img_features @ gt_features.T).squeeze())
    clip_img_gen_sim = float((img_features @ gen_features.T).squeeze())
    clip_text_gen_sim = float((gt_features @ gen_features.T).squeeze())
    clip_sim_gap = clip_img_gen_sim - clip_img_text_sim

    return {
        "clip_img_text_sim": clip_img_text_sim,
        "clip_img_gen_sim": clip_img_gen_sim,
        "clip_text_gen_sim": clip_text_gen_sim,
        "clip_sim_gap": clip_sim_gap,
    }


# =============================================================================
# Combined Feature Extraction
# =============================================================================
def extract_all_features(vlm_model, processor, clip_model, clip_tokenizer,
                         clip_preprocess, dataset, device, desc="Extracting",
                         max_samples=None):
    """
    Extract all M⁴I-adapted features for a dataset split.
    Each sample gets: MB-M⁴I metrics + FB-M⁴I CLIP features + auxiliary.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    records = []

    for i, example in enumerate(tqdm(dataset, desc=desc, total=max_samples)):
        if max_samples and i >= max_samples:
            break

        try:
            doc_id = example["id"]
            text = example["text"]
            label = example.get("is_member", None)
            image = extract_image(example)

            # Step 1: Generate text from finetuned VLM
            generated = generate_text(vlm_model, processor, image, device)

            # Step 2: MB-M⁴I features (metric-based comparison)
            metric_feats = compute_metric_features(text, generated, scorer)

            # Step 3: FB-M⁴I features (CLIP-based similarity)
            clip_feats = compute_clip_features(
                image, text, generated,
                clip_model, clip_tokenizer, clip_preprocess, device
            )

            # Combine
            record = {
                "id": doc_id,
                "is_member": label,
                **metric_feats,
                **clip_feats,
                "gen_length": len(generated.split()),
            }
            records.append(record)

        except Exception as e:
            print(f"  ⚠ Error on sample {example.get('id', i)}: {e}")
            continue

    return pd.DataFrame(records)


# =============================================================================
# Main
# =============================================================================
def get_best_dtype(device):
    """Pick the best dtype for the given device."""
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16     # fallback for older GPUs
    return torch.float32         # CPU / MPS


def log_gpu_memory(tag=""):
    """Print current VRAM usage (CUDA only)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [GPU {tag}] allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="M⁴I-Adapted MIA Attack (CLIP + Metric-Based)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run on small subset (~50 per split)"
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Re-extract features, ignoring cache"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device: cuda, mps, or cpu (auto-detected if omitted)"
    )
    args = parser.parse_args()

    max_samples = 50 if args.debug else None
    suffix = "_debug" if args.debug else ""

    # ---- Device ----
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = get_best_dtype(device)
    print(f"Device: {device}  |  dtype: {dtype}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        log_gpu_memory("startup")

    # ---- Load SmolVLM (finetuned target model) ----
    print("\nLoading finetuned SmolVLM...")
    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}

    # Use flash_attention_2 on CUDA (faster), sdpa elsewhere
    attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"
    try:
        vlm_model = (
            SmolVLMForConditionalGeneration.from_pretrained(
                FINETUNED_MODEL_ID,
                torch_dtype=dtype,
                _attn_implementation=attn_impl,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
    except Exception:
        # flash_attention_2 may not be installed; fall back to sdpa
        print("  (flash_attention_2 not available, using sdpa)")
        vlm_model = (
            SmolVLMForConditionalGeneration.from_pretrained(
                FINETUNED_MODEL_ID,
                torch_dtype=dtype,
                _attn_implementation="sdpa",
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
    log_gpu_memory("after VLM load")

    # ---- Load CLIP ----
    print("Loading CLIP (ViT-B/32)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    clip_model = clip_model.to(dtype).eval()   # match VLM dtype for consistency
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    log_gpu_memory("after CLIP load")

    # ---- Load Dataset (streaming) ----
    print("\nLoading dataset...")
    ds_train = load_dataset(DATASET_ID, split="train", streaming=True)
    ds_val = load_dataset(DATASET_ID, split="validation", streaming=True)
    ds_test = load_dataset(DATASET_ID, split="test", streaming=True)

    # ---- Feature Extraction with Caching ----
    os.makedirs(CACHE_DIR, exist_ok=True)

    splits = {
        "train": ds_train,
        "val": ds_val,
        "test": ds_test,
    }
    dfs = {}

    for split_name, ds in splits.items():
        cache_path = os.path.join(CACHE_DIR, f"{split_name}_m4i{suffix}.json")

        if os.path.exists(cache_path) and not args.recompute:
            print(f"\nLoading cached {split_name} features from {cache_path}")
            dfs[split_name] = pd.read_json(cache_path)
        else:
            print(f"\n=== Extracting {split_name} features ===")
            df = extract_all_features(
                vlm_model, processor,
                clip_model, clip_tokenizer, clip_preprocess,
                ds, device,
                desc=split_name.capitalize(),
                max_samples=max_samples,
            )
            df.to_json(cache_path, orient="records", indent=2)
            print(f"  → Cached to {cache_path}")
            dfs[split_name] = df

    df_train = dfs["train"]
    df_val = dfs["val"]
    df_test = dfs["test"]

    # Free VLM and CLIP memory before classifier training
    del vlm_model, clip_model
    if device == "cuda":
        torch.cuda.empty_cache()
        log_gpu_memory("after cleanup")

    # ---- Train XGBoost Classifier ----
    print("\n" + "=" * 60)
    print(" Training Meta-Classifier")
    print("=" * 60)

    # Check for missing features
    available_feats = [c for c in FEATURE_COLS if c in df_train.columns]
    if len(available_feats) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_feats)
        print(f"  ⚠ Missing features (will be skipped): {missing}")
    feat_cols = available_feats

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feat_cols].values)
    y_train = df_train["is_member"].values

    if XGBClassifier is not None:
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
        print("  Classifier: GradientBoosting (fallback)")
        clf = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    clf.fit(X_train, y_train)

    # ---- Evaluate on Validation ----
    print("\n" + "=" * 60)
    print(" Validation Results — M⁴I-Adapted Attack")
    print("=" * 60)

    X_val = scaler.transform(df_val[feat_cols].values)
    y_val = df_val["is_member"].values
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, val_probs)
    val_tpr = tpr_at_fpr(y_val, val_probs)
    print(f"  AUC:          {val_auc:.4f}")
    print(f"  TPR@FPR=0.1:  {val_tpr:.4f}")

    print("\n  Feature Importances:")
    importances = sorted(
        zip(feat_cols, clf.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances:
        print(f"    {name:25s}: {imp:.4f}")

    # ---- Per-feature AUC (for analysis) ----
    print("\n  Per-feature AUC (univariate):")
    for col in feat_cols:
        try:
            auc_pos = roc_auc_score(y_val, df_val[col].values)
            auc_neg = roc_auc_score(y_val, -df_val[col].values)
            best_auc = max(auc_pos, auc_neg)
            direction = "+" if auc_pos >= auc_neg else "-"
            print(f"    {col:25s}: {best_auc:.4f} ({direction})")
        except ValueError:
            print(f"    {col:25s}: N/A")

    # ---- Generate Test Submission ----
    print("\n=== Generating Kaggle Submission ===")

    X_test = scaler.transform(df_test[feat_cols].values)
    test_probs = clf.predict_proba(X_test)[:, 1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub_path = os.path.join(OUTPUT_DIR, "m4i_clip_submission.csv")
    submission = pd.DataFrame({"id": df_test["id"], "is_member": test_probs})
    submission.to_csv(sub_path, index=False)
    print(f"  Saved to {sub_path}")
    print(f"  Score stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    # Save detailed features
    detail_path = os.path.join(OUTPUT_DIR, "m4i_clip_features.csv")
    df_test[["id"] + feat_cols].to_csv(detail_path, index=False)
    print(f"  Feature details saved to {detail_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
