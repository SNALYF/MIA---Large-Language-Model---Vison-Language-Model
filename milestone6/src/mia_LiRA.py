"""
Ultimate Hybrid MIA — All Signals Combined
=============================================

Combines every viable signal into one meta-classifier:

  A) GENERATION + METRICS (the ~0.80 AUC foundation)
     Generate text from finetuned VLM → compare to ground truth via
     ROUGE-L, ROUGE-1, ROUGE-2, BLEU-1, BLEU-2, token overlap.

  B) CLIP EMBEDDINGS
     Cosine similarities between image, ground-truth text, and
     generated text in CLIP space.

  C) LiRA CROSS-MODEL FEATURES
     Forward pass through BOTH finetuned and base model → compute
     loss_ratio, loss_diff, perplexity_ratio, min_k_diff, etc.
     Orthogonal signal that captures "how much did finetuning change
     this sample's loss?"

  D) GENERATION CONSISTENCY
     Generate 2 additional times with sampling at T=1.0 → measure
     pairwise similarity. Members produce more consistent outputs.

All features → XGBoost (or GBC fallback).

Usage:
    python mia_ultimate.py                # full run
    python mia_ultimate.py --debug        # ~50 samples per split
    python mia_ultimate.py --recompute    # ignore cache
    python mia_ultimate.py --device cuda
"""

import os
import io
import zlib
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from rouge_score import rouge_scorer
except ImportError:
    raise ImportError("pip install rouge-score")

try:
    import open_clip
except ImportError:
    raise ImportError("pip install open_clip_torch")

try:
    from xgboost import XGBClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBClassifier = None
    print("WARNING: xgboost not found → sklearn GBC fallback")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    sentence_bleu = None
    print("WARNING: nltk not found → BLEU features disabled")

from transformers import AutoProcessor, SmolVLMForConditionalGeneration

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────
FINETUNED_MODEL_ID = "UBC-SLIME/colx_585_vlm"
BASE_MODEL_ID      = "HuggingFaceTB/SmolVLM-256M-Instruct"
BASE_PROCESSOR_ID  = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASET_ID         = "UBC-SLIME/colx585_group_project_data"

CLIP_MODEL_NAME  = "ViT-B-32"
CLIP_PRETRAINED  = "openai"

MAX_GEN_TOKENS    = 300
GENERATION_PROMPT = "Describe this image."
MIN_K_PERCENT     = 20
CONSISTENCY_REPS  = 2   # Extra sampled generations for consistency

CACHE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultimate_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════
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


def parse_text(text):
    parts = text.split("\n", 1)
    return parts[0], (parts[1] if len(parts) > 1 else text)


def tpr_at_fpr(y_true, y_score, target_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    return tpr[idx[-1]] if len(idx) > 0 else 0.0


def get_best_dtype(device):
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def log_gpu(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        print(f"  [GPU {tag}] alloc={a:.2f}GB")


def rouge_f1(scorer, ref, hyp):
    if not ref.strip() or not hyp.strip():
        return {"rougeL": 0.0, "rouge1": 0.0, "rouge2": 0.0}
    scores = scorer.score(ref, hyp)
    return {
        "rougeL": scores["rougeL"].fmeasure,
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
    }


# ═════════════════════════════════════════════════════════════════════════
# Signal A: VLM Text Generation
# ═════════════════════════════════════════════════════════════════════════
def generate_text(model, processor, image, device, temperature=1.0,
                  do_sample=True):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": GENERATION_PROMPT},
        ]},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(model_dtype) if v.is_floating_point() else v
        for k, v in inputs.items()
    }

    gen_kwargs = dict(max_new_tokens=MAX_GEN_TOKENS, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    return processor.tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()


# ═════════════════════════════════════════════════════════════════════════
# Signal A: Metric-Based Features (MB-M⁴I)
# ═════════════════════════════════════════════════════════════════════════
def compute_metric_features(gt_text, generated_text, scorer):
    _, gt_answer = parse_text(gt_text)

    rouges = rouge_f1(scorer, gt_answer, generated_text)

    gt_tokens = set(gt_answer.lower().split())
    gen_tokens = set(generated_text.lower().split())
    if len(gt_tokens) == 0 or len(gen_tokens) == 0:
        token_overlap = 0.0
    else:
        token_overlap = len(gt_tokens & gen_tokens) / len(gt_tokens | gen_tokens)

    bleu_1, bleu_2 = 0.0, 0.0
    if sentence_bleu is not None:
        smooth = SmoothingFunction().method1
        ref = gt_answer.lower().split()
        hyp = generated_text.lower().split()
        if len(ref) > 0 and len(hyp) > 0:
            try:
                bleu_1 = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0),
                                       smoothing_function=smooth)
                bleu_2 = sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0),
                                       smoothing_function=smooth)
            except Exception:
                pass

    return {
        "rouge_l": rouges["rougeL"],
        "rouge_1": rouges["rouge1"],
        "rouge_2": rouges["rouge2"],
        "token_overlap": token_overlap,
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
    }


# ═════════════════════════════════════════════════════════════════════════
# Signal B: CLIP Embedding Features (FB-M⁴I)
# ═════════════════════════════════════════════════════════════════════════
def compute_clip_features(image, gt_text, generated_text,
                          clip_model, clip_tokenizer, clip_preprocess, device):
    _, gt_answer = parse_text(gt_text)
    gt_clip = gt_answer[:300]
    gen_clip = generated_text[:300]

    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = clip_model.encode_image(image_input)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    gt_tokens = clip_tokenizer([gt_clip]).to(device)
    gen_tokens = clip_tokenizer([gen_clip]).to(device)

    with torch.no_grad():
        gt_features = clip_model.encode_text(gt_tokens)
        gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)
        gen_features = clip_model.encode_text(gen_tokens)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

    clip_img_text = float((img_features @ gt_features.T).squeeze())
    clip_img_gen = float((img_features @ gen_features.T).squeeze())
    clip_text_gen = float((gt_features @ gen_features.T).squeeze())
    clip_sim_gap = clip_img_gen - clip_img_text

    return {
        "clip_img_text_sim": clip_img_text,
        "clip_img_gen_sim": clip_img_gen,
        "clip_text_gen_sim": clip_text_gen,
        "clip_sim_gap": clip_sim_gap,
    }


# ═════════════════════════════════════════════════════════════════════════
# Signal C: LiRA — Token-Level Features from ONE Model
# ═════════════════════════════════════════════════════════════════════════
def compute_token_features(model, processor, example, device, suffix=""):
    image = extract_image(example)
    question, answer = parse_text(example["text"])

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(model_dtype) if v.is_floating_point() else v
        for k, v in inputs.items()
    }

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    mask = shift_labels != -100
    token_lp = token_log_probs[mask]

    if token_lp.numel() == 0:
        return None

    token_losses = -token_lp
    mean_loss = token_losses.mean().item()
    perplexity = np.exp(min(mean_loss, 100))

    k = max(1, int(len(token_lp) * MIN_K_PERCENT / 100))
    sorted_lp, _ = token_lp.sort()
    min_k_prob = sorted_lp[:k].mean().item()

    max_token_loss = token_losses.max().item()
    std_token_loss = token_losses.std().item() if len(token_losses) > 1 else 0.0

    text_bytes = answer.encode("utf-8")
    zlib_len = len(zlib.compress(text_bytes)) if len(text_bytes) > 0 else 1
    zlib_ratio = mean_loss / zlib_len

    s = suffix
    return {
        f"loss{s}": mean_loss,
        f"perplexity{s}": perplexity,
        f"min_k_prob{s}": min_k_prob,
        f"max_token_loss{s}": max_token_loss,
        f"std_token_loss{s}": std_token_loss,
        f"zlib_ratio{s}": zlib_ratio,
    }


# ═════════════════════════════════════════════════════════════════════════
# Signal C: LiRA Cross-Model Features
# ═════════════════════════════════════════════════════════════════════════
def compute_lira_features(ft_feats, base_feats):
    return {
        "loss_ratio": base_feats["loss_base"] / (ft_feats["loss_ft"] + 1e-8),
        "loss_diff": base_feats["loss_base"] - ft_feats["loss_ft"],
        "min_k_diff": ft_feats["min_k_prob_ft"] - base_feats["min_k_prob_base"],
        "perplexity_ratio": base_feats["perplexity_base"] / (ft_feats["perplexity_ft"] + 1e-8),
        "max_loss_diff": base_feats["max_token_loss_base"] - ft_feats["max_token_loss_ft"],
        "std_loss_diff": base_feats["std_token_loss_base"] - ft_feats["std_token_loss_ft"],
        "zlib_ratio_diff": base_feats["zlib_ratio_base"] - ft_feats["zlib_ratio_ft"],
        "norm_loss_improvement": (
            (base_feats["loss_base"] - ft_feats["loss_ft"])
            / (base_feats["loss_base"] + 1e-8)
        ),
    }


# ═════════════════════════════════════════════════════════════════════════
# Signal D: Generation Consistency
# ═════════════════════════════════════════════════════════════════════════
def compute_consistency_features(ft_model, processor, image, device,
                                 greedy_gen, scorer):
    extra_gens = []
    for _ in range(CONSISTENCY_REPS):
        g = generate_text(ft_model, processor, image, device,
                          temperature=1.0, do_sample=True)
        extra_gens.append(g)

    all_gens = [greedy_gen] + extra_gens

    pairwise = []
    for i in range(len(all_gens)):
        for j in range(i + 1, len(all_gens)):
            r = rouge_f1(scorer, all_gens[i], all_gens[j])
            pairwise.append(r["rougeL"])

    return {
        "gen_consistency_mean": np.mean(pairwise) if pairwise else 0.0,
        "gen_consistency_std": np.std(pairwise) if len(pairwise) > 1 else 0.0,
    }


# ═════════════════════════════════════════════════════════════════════════
# Combined Extraction
# ═════════════════════════════════════════════════════════════════════════
def extract_all_features(ft_model, base_model, processor,
                         clip_model, clip_tokenizer, clip_preprocess,
                         dataset, device, scorer, max_samples=None):
    records = []

    for i, example in enumerate(tqdm(dataset, desc="Extracting", total=max_samples)):
        if max_samples and i >= max_samples:
            break

        try:
            doc_id = example["id"]
            label = example.get("is_member", None)
            image = extract_image(example)

            record = {"id": doc_id, "is_member": label}

            # ── Signal A: Generate text + metric features ──
            greedy_gen = generate_text(ft_model, processor, image, device,
                                       do_sample=False)
            metric_feats = compute_metric_features(
                example["text"], greedy_gen, scorer
            )
            record.update(metric_feats)
            record["gen_length"] = len(greedy_gen.split())

            # ── Signal B: CLIP features ──
            clip_feats = compute_clip_features(
                image, example["text"], greedy_gen,
                clip_model, clip_tokenizer, clip_preprocess, device
            )
            record.update(clip_feats)

            # ── Signal C: LiRA (finetuned vs base forward pass) ──
            ft_feats = compute_token_features(
                ft_model, processor, example, device, suffix="_ft"
            )
            base_feats = compute_token_features(
                base_model, processor, example, device, suffix="_base"
            )

            if ft_feats is not None and base_feats is not None:
                record.update(ft_feats)
                record.update(base_feats)
                lira_feats = compute_lira_features(ft_feats, base_feats)
                record.update(lira_feats)

            # ── Signal D: Generation consistency ──
            consistency_feats = compute_consistency_features(
                ft_model, processor, image, device, greedy_gen, scorer
            )
            record.update(consistency_feats)

            records.append(record)

        except Exception as e:
            print(f"  ⚠ Error on sample {example.get('id', i)}: {e}")
            continue

    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════
# Feature Columns
# ═════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # A: Metric-based
    "rouge_l", "rouge_1", "rouge_2", "token_overlap",
    "bleu_1", "bleu_2", "gen_length",
    # B: CLIP
    "clip_img_text_sim", "clip_img_gen_sim", "clip_text_gen_sim", "clip_sim_gap",
    # C: LiRA — finetuned model
    "loss_ft", "perplexity_ft", "min_k_prob_ft",
    "max_token_loss_ft", "std_token_loss_ft", "zlib_ratio_ft",
    # C: LiRA — base model
    "loss_base", "perplexity_base", "min_k_prob_base",
    "max_token_loss_base", "std_token_loss_base", "zlib_ratio_base",
    # C: LiRA — cross-model
    "loss_ratio", "loss_diff", "min_k_diff", "perplexity_ratio",
    "max_loss_diff", "std_loss_diff", "zlib_ratio_diff",
    "norm_loss_improvement",
    # D: Consistency
    "gen_consistency_mean", "gen_consistency_std",
]


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Ultimate Hybrid MIA")
    parser.add_argument("--debug", action="store_true",
                        help="~50 samples per split")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cache")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    max_samples = 50 if args.debug else None
    suffix = "_debug" if args.debug else ""

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

    # ── Load processor ──
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.max_image_size = {"longest_edge": 512}

    # ── Load finetuned model ──
    print("Loading FINETUNED SmolVLM...")
    attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"
    try:
        ft_model = SmolVLMForConditionalGeneration.from_pretrained(
            FINETUNED_MODEL_ID, torch_dtype=dtype,
            _attn_implementation=attn_impl, trust_remote_code=True,
        ).to(device).eval()
    except Exception:
        print("  (flash_attention_2 unavailable → sdpa)")
        ft_model = SmolVLMForConditionalGeneration.from_pretrained(
            FINETUNED_MODEL_ID, torch_dtype=dtype,
            _attn_implementation="sdpa", trust_remote_code=True,
        ).to(device).eval()
    log_gpu("finetuned")

    # ── Load base model ──
    print("Loading BASE SmolVLM...")
    try:
        base_model = SmolVLMForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, torch_dtype=dtype,
            _attn_implementation=attn_impl, trust_remote_code=True,
        ).to(device).eval()
    except Exception:
        base_model = SmolVLMForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, torch_dtype=dtype,
            _attn_implementation="sdpa", trust_remote_code=True,
        ).to(device).eval()
    log_gpu("base")

    # ── Load CLIP ──
    print("Loading CLIP (ViT-B/32)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    clip_model = clip_model.to(dtype).eval()
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    log_gpu("CLIP")

    # ── ROUGE scorer ──
    scorer = rouge_scorer.RougeScorer(
        ["rougeL", "rouge1", "rouge2"], use_stemmer=True
    )

    # ── Load dataset ──
    print("\nLoading dataset...")
    splits = {
        "train": load_dataset(DATASET_ID, split="train", streaming=True),
        "val":   load_dataset(DATASET_ID, split="validation", streaming=True),
        "test":  load_dataset(DATASET_ID, split="test", streaming=True),
    }

    # ── Feature extraction ──
    os.makedirs(CACHE_DIR, exist_ok=True)
    dfs = {}

    for split_name, ds in splits.items():
        cache_path = os.path.join(CACHE_DIR, f"{split_name}{suffix}.json")

        if os.path.exists(cache_path) and not args.recompute:
            print(f"\nLoading cached {split_name} from {cache_path}")
            dfs[split_name] = pd.read_json(cache_path)
        else:
            print(f"\n=== Extracting {split_name} features ===")
            df = extract_all_features(
                ft_model, base_model, processor,
                clip_model, clip_tokenizer, clip_preprocess,
                ds, device, scorer,
                max_samples=max_samples,
            )
            df.to_json(cache_path, orient="records", indent=2)
            print(f"  → Cached ({len(df)} samples)")
            dfs[split_name] = df

    df_train = dfs["train"]
    df_val = dfs["val"]
    df_test = dfs["test"]

    # Free models
    del ft_model, base_model, clip_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Select available features ──
    feat_cols = [c for c in FEATURE_COLS if c in df_train.columns]
    missing = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        print(f"\n  ⚠ Missing features: {missing}")
    print(f"\n  Using {len(feat_cols)} features")

    # Clean data
    for df in [df_train, df_val, df_test]:
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
        df[feat_cols] = df[feat_cols].fillna(0)

    # ── Train classifier ──
    print("\n" + "=" * 60)
    print(" Training Meta-Classifier")
    print("=" * 60)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feat_cols].values)
    y_train = df_train["is_member"].values

    if XGBClassifier is not None:
        print("  Classifier: XGBoost")
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        print("  Classifier: GradientBoosting (fallback)")
        clf = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )

    clf.fit(X_train, y_train)

    # ── Evaluate ──
    print("\n" + "=" * 60)
    print(" Validation Results — Ultimate Hybrid MIA")
    print("=" * 60)

    X_val = scaler.transform(df_val[feat_cols].values)
    y_val = df_val["is_member"].values
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, val_probs)
    val_tpr = tpr_at_fpr(y_val, val_probs)
    print(f"  AUC:          {val_auc:.4f}")
    print(f"  TPR@FPR=0.1:  {val_tpr:.4f}")

    # Feature importances
    print("\n  Feature Importances (top 15):")
    importances = sorted(
        zip(feat_cols, clf.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances[:15]:
        print(f"    {name:30s}: {imp:.4f}")

    # Per-feature AUC
    print("\n  Per-feature AUC (univariate):")
    for col in feat_cols:
        try:
            auc_pos = roc_auc_score(y_val, df_val[col].values)
            auc_neg = roc_auc_score(y_val, -df_val[col].values)
            best = max(auc_pos, auc_neg)
            d = "+" if auc_pos >= auc_neg else "-"
            print(f"    {col:30s}: {best:.4f} ({d})")
        except ValueError:
            print(f"    {col:30s}: N/A")

    # ── Signal family ablation ──
    print("\n  Signal Family Ablation:")
    families = {
        "A (Metric)": ["rouge_l", "rouge_1", "rouge_2", "token_overlap",
                        "bleu_1", "bleu_2", "gen_length"],
        "B (CLIP)": ["clip_img_text_sim", "clip_img_gen_sim",
                      "clip_text_gen_sim", "clip_sim_gap"],
        "A+B (Colleague)": ["rouge_l", "rouge_1", "rouge_2", "token_overlap",
                             "bleu_1", "bleu_2", "gen_length",
                             "clip_img_text_sim", "clip_img_gen_sim",
                             "clip_text_gen_sim", "clip_sim_gap"],
        "C (LiRA only)": ["loss_ratio", "loss_diff", "min_k_diff",
                           "perplexity_ratio", "max_loss_diff",
                           "std_loss_diff", "zlib_ratio_diff",
                           "norm_loss_improvement"],
        "D (Consistency)": ["gen_consistency_mean", "gen_consistency_std"],
    }

    for family_name, family_cols in families.items():
        avail = [c for c in family_cols if c in df_train.columns]
        if not avail:
            continue
        try:
            sc = StandardScaler()
            Xtr = sc.fit_transform(df_train[avail].values)
            Xva = sc.transform(df_val[avail].values)
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(Xtr, y_train)
            probs = lr.predict_proba(Xva)[:, 1]
            auc = roc_auc_score(y_val, probs)
            print(f"    {family_name:25s}: AUC = {auc:.4f}")
        except Exception as e:
            print(f"    {family_name:25s}: Error ({e})")

    # ── Test submission ──
    print("\n=== Generating Kaggle Submission ===")
    X_test = scaler.transform(df_test[feat_cols].values)
    test_probs = clf.predict_proba(X_test)[:, 1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub_path = os.path.join(OUTPUT_DIR, "ultimate_submission.csv")
    submission = pd.DataFrame({"id": df_test["id"], "is_member": test_probs})
    submission.to_csv(sub_path, index=False)
    print(f"  Saved to {sub_path}")
    print(f"  Stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")
    print(f"  Rows: {len(submission)}")

    # Save features for analysis
    feat_path = os.path.join(OUTPUT_DIR, "ultimate_features.csv")
    df_test[["id"] + feat_cols].to_csv(feat_path, index=False)
    print(f"  Features saved to {feat_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
