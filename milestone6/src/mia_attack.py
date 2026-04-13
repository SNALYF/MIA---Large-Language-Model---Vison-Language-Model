"""
Membership Inference Attack for Fine-tuned VLM
================================================
Task: Detect whether an (image, caption) pair was seen during VLM fine-tuning.
  - Label 1: seen image + seen caption (joint member)
  - Label 0: any other combination

Three-layer attack strategy:
  1. Loss Ratio          — fine-tuned loss vs. base model loss
  2. Caption Contrast    — original caption loss vs. shuffled caption loss
  3. Image Corruption    — clean image loss vs. corrupted image loss

Usage:
  - Fill in the config section at the top with your model paths / data paths.
  - Run extract_all_features() on train/val/test splits.
  - Run train_and_predict() to get submission scores.
"""

from google.colab import drive

drive.mount("/content/drive")

import os
import io
import json
import random
import argparse
from threading import Thread
from queue import Queue
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ─────────────────────────────────────────────
#  CONFIG — edit these to match your setup
# ─────────────────────────────────────────────
CONFIG = {
    # Paths to fine-tuned and base (pre-fine-tune) checkpoints
    "finetuned_model_path": "UBC-SLIME/colx_585_vlm",
    "base_model_path": "HuggingFaceTB/SmolVLM-256M-Instruct",  # set to None to skip loss ratio
    # HuggingFace dataset — each example has: id, text, is_member, image
    "dataset_id": "UBC-SLIME/colx585_group_project_data",
    # Attack hyperparameters
    "n_shuffles": 3,  # how many negative captions to average over
    "n_corruptions": 1,  # how many corrupted images to average over
    "corruption_radius": 2,  # GaussianBlur radius for image corruption
    "noise_std": 0.05,  # Gaussian noise std (relative to [0,1] range)
    "batch_size": 128,
    "debug_max": 500,  # samples per split in debug mode; set to None for full run
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    # 修改前：
    # "cache_dir": os.path.join(os.path.dirname(__file__), "features_cache"),
    # 修改后：将其指向你的 Google Drive 中的某个文件夹，例如 "VLM_MIA_Project"
    "cache_dir": "/content/drive/MyDrive/VLM_MIA_Project/features_cache",
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])


# ─────────────────────────────────────────────
#  IMAGE EXTRACTION HELPER
# ─────────────────────────────────────────────


def extract_image(example):
    """Extract PIL Image from a HuggingFace dataset example."""
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


# ─────────────────────────────────────────────
#  MODEL LOADING  (adapt to your VLM API)
# ─────────────────────────────────────────────


def load_model(model_path: str):
    device = CONFIG["device"]
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_path)

    # ======= 新增这三行来防止 Batch Padding 报错 =======
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    # =================================================

    model = (
        SmolVLMForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    return model, processor


# ─────────────────────────────────────────────
#  CORE: compute token-level cross-entropy loss
#  for one (image, caption) pair
# ─────────────────────────────────────────────


def _prepare_caption(caption: str) -> str:
    """Extract assistant text from caption and truncate.

    Expects text field to have format: '<prefix>\n<assistant response>'.
    Strips common role prefixes (e.g. 'Assistant:') if present.
    """
    parts = caption.split("\n", 1)
    assistant_text = parts[1].strip() if len(parts) > 1 else caption.strip()
    for prefix in ("Assistant:", "ASSISTANT:", "assistant:", "A:"):
        if assistant_text.startswith(prefix):
            assistant_text = assistant_text[len(prefix) :].strip()
            break
    return " ".join(assistant_text.split()[:300])


def _compute_prompt_only_len(processor, image: Image.Image) -> int:
    """Compute the token length of the prompt-only template using a real image.

    Uses processor(..., images=[image]) so that image tokens (which depend on
    the image's tiling/resolution) are counted accurately. Called once per
    batch using the first image, since all images in a batch share the same
    tiling after padding.
    """
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]
    prompt = processor.apply_chat_template(msgs, tokenize=False)
    return processor(text=prompt, images=[image], return_tensors="pt")[
        "input_ids"
    ].shape[1]


def compute_loss(model, processor, image: Image.Image, caption: str) -> float:
    """
    Returns the mean cross-entropy loss on the assistant (caption) tokens only.
    Uses SmolVLM chat template: user asks "Describe this image.", assistant responds with caption.
    Lower loss → model is more confident about this pair → likely a member.
    """
    device = CONFIG["device"]
    assistant_text = _prepare_caption(caption)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False)

    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    prompt_only_len = _compute_prompt_only_len(processor, image)
    if device == "cuda":
        inputs = {
            k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
            for k, v in inputs.items()
        }

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        logits = model(**inputs).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    assistant_start = max(0, prompt_only_len - 1)
    assistant_losses = per_token_loss[assistant_start:]

    if len(assistant_losses) == 0:
        return per_token_loss.mean().item()

    return assistant_losses.mean().item()


def compute_loss_batch(model, processor, images: list, captions: list) -> list:
    """
    Batch version of compute_loss. Processes multiple (image, caption) pairs
    with automatic sub-batching to prevent OOM.
    """
    if len(images) == 0:
        return []
    if len(images) == 1:
        return [compute_loss(model, processor, images[0], captions[0])]

    # Sub-batch to prevent OOM on large mega-batches
    max_batch = CONFIG["batch_size"]
    if len(images) > max_batch:
        all_results = []
        for start in range(0, len(images), max_batch):
            end = min(start + max_batch, len(images))
            all_results.extend(
                _compute_loss_batch_core(
                    model, processor, images[start:end], captions[start:end]
                )
            )
        return all_results

    return _compute_loss_batch_core(model, processor, images, captions)


def _compute_loss_batch_core(model, processor, images: list, captions: list) -> list:
    """Single-batch forward pass for compute_loss_batch."""
    device = CONFIG["device"]
    prompt_only_len = _compute_prompt_only_len(processor, images[0])

    # Build prompts for each pair
    prompts = []
    for caption in captions:
        assistant_text = _prepare_caption(caption)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
        prompts.append(processor.apply_chat_template(messages, tokenize=False))

    # Idefics3 架构要求 batched images 的格式必须是 list of lists
    batched_images = [[img] for img in images]

    # Process batch with padding
    inputs = processor(
        text=prompts, images=batched_images, return_tensors="pt", padding=True
    ).to(device)

    if device == "cuda":
        inputs = {
            k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
            for k, v in inputs.items()
        }

    input_ids = inputs["input_ids"]  # (B, seq_len)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    with torch.no_grad():
        logits = model(**inputs).logits  # (B, seq_len, vocab)

    # Per-token cross-entropy for each item in batch
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(
        shift_labels.size()
    )  # (B, seq_len-1)

    # Compute per-example mean loss on assistant tokens only
    assistant_start = max(0, prompt_only_len - 1)
    results = []
    for i in range(len(images)):
        assistant_losses = per_token_loss[i, assistant_start:]
        assistant_mask = shift_mask[i, assistant_start:]
        masked_losses = assistant_losses * assistant_mask
        n_tokens = assistant_mask.sum()
        if n_tokens == 0:
            results.append(per_token_loss[i].mean().item())
        else:
            results.append((masked_losses.sum() / n_tokens).item())

    return results


# ─────────────────────────────────────────────
#  IMAGE HIDDEN STATE CACHING
#  Run vision encoder + connector ONCE per image,
#  then reuse for shuffled caption variants.
# ─────────────────────────────────────────────


def extract_image_hidden_states_batch(model, processor, images: list):
    """
    Run vision encoder + connector for a batch of images.
    Returns (B, n_img_tokens, hidden_dim) tensor on GPU.
    Call this ONCE per unique set of images; pass the result to
    compute_loss_batch_with_hidden to skip redundant vision encoder runs.

    SmolVLM processor returns pixel_values as (B, num_sub, C, H, W).
    The vision model forward expects (N, C, H, W), so we flatten the
    sub-image dimension first and reshape the output back.
    """
    device = CONFIG["device"]

    dummy_prompts = [
        processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": "x"}],
                }
            ],
            tokenize=False,
        )
        for _ in images
    ]
    batched_images = [[img] for img in images]
    raw = processor(
        text=dummy_prompts, images=batched_images, return_tensors="pt", padding=True
    )

    # pixel_values: (B, num_sub, C, H, W) when tiling is used, else (B, C, H, W)
    pixel_values = raw["pixel_values"]

    if pixel_values.dim() == 5:
        B, num_sub = pixel_values.shape[:2]
        # Flatten sub-image dim: (B*num_sub, C, H, W)
        pv_flat = pixel_values.reshape(B * num_sub, *pixel_values.shape[2:]).to(device)
    else:
        # No tiling: (B, C, H, W) — treat as B images each with 1 sub-image
        B, num_sub = pixel_values.shape[0], 1
        pv_flat = pixel_values.to(device)

    if device == "cuda":
        pv_flat = pv_flat.to(torch.bfloat16)

    with torch.no_grad():
        vision_kwargs = {"pixel_values": pv_flat}
        # Do NOT pass pixel_attention_mask to the vision model — the processor returns
        # a pixel-space mask while the vision encoder expects a patch-space mask;
        # passing the wrong shape causes an astronomical allocation error. All images
        # are valid so full attention (no mask) is correct.
        # hidden: (B*num_sub, n_patches, dim)
        hidden = model.model.vision_model(**vision_kwargs).last_hidden_state
        # connector: (B*num_sub, n_tokens, dim)
        hidden = model.model.connector(hidden)
        # reshape back: (B, num_sub * n_tokens, dim)
        hidden = hidden.reshape(B, -1, hidden.shape[-1])

    return hidden  # (B, n_total_img_tokens, dim)


def _compute_loss_batch_core_with_hidden(
    model, processor, image_hidden_states, images: list, captions: list
) -> list:
    """
    Single-batch forward pass using pre-computed image hidden states.
    Passes image_hidden_states directly to the model → vision encoder is skipped.
    images: still needed so the processor produces input_ids with the correct
            number of image token placeholders (CPU-only work, no GPU vision pass).
    prompt_only_len is computed from images[0] to correctly account for the
    actual number of image tokens in the input sequence.
    """
    device = CONFIG["device"]
    prompt_only_len = _compute_prompt_only_len(processor, images[0])

    prompts = []
    for caption in captions:
        assistant_text = _prepare_caption(caption)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
        prompts.append(processor.apply_chat_template(messages, tokenize=False))

    batched_images = [[img] for img in images]
    # CPU only — we need input_ids with the right image-token count; pixel_values are discarded
    raw = processor(
        text=prompts, images=batched_images, return_tensors="pt", padding=True
    )
    input_ids = raw["input_ids"].to(device)
    attention_mask = (
        raw["attention_mask"].to(device) if "attention_mask" in raw else None
    )

    if device == "cuda":
        image_hidden_states = image_hidden_states.to(torch.bfloat16)

    with torch.no_grad():
        # image_hidden_states → vision encoder inside the model is bypassed
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_hidden_states=image_hidden_states,
        )
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = (
        attention_mask[:, 1:].float()
        if attention_mask is not None
        else torch.ones_like(shift_labels).float()
    )

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    assistant_start = max(0, prompt_only_len - 1)
    results = []
    for i in range(len(captions)):
        al = per_token_loss[i, assistant_start:]
        am = shift_mask[i, assistant_start:]
        n = am.sum()
        results.append(
            (al * am).sum().item() / n.item()
            if n > 0
            else per_token_loss[i].mean().item()
        )
    return results


def _safe_stack(hs_list: list) -> torch.Tensor:
    """Stack hidden state tensors, padding to the same length if shapes differ."""
    shapes = [h.shape[0] for h in hs_list]
    if len(set(shapes)) == 1:
        return torch.stack(hs_list, dim=0)
    max_n = max(shapes)
    padded = [
        F.pad(h, (0, 0, 0, max_n - h.shape[0])) if h.shape[0] < max_n else h
        for h in hs_list
    ]
    return torch.stack(padded, dim=0)


def compute_loss_batch_with_hidden(
    model, processor, image_hidden_states, images: list, captions: list
) -> list:
    """
    Chunked wrapper around _compute_loss_batch_core_with_hidden.
    image_hidden_states: list of (n_img_tokens, dim) tensors  OR  (B, n, dim) tensor.
    Handles variable image token counts by padding hidden states to the same length.
    """
    if not captions:
        return []

    batch_size = CONFIG["batch_size"]

    if isinstance(image_hidden_states, torch.Tensor):
        hs_list = [image_hidden_states[i] for i in range(image_hidden_states.shape[0])]
    else:
        hs_list = list(image_hidden_states)

    if len(captions) <= batch_size:
        return _compute_loss_batch_core_with_hidden(
            model,
            processor,
            _safe_stack(hs_list),
            images,
            captions,
        )

    all_results = []
    for start in range(0, len(captions), batch_size):
        end = min(start + batch_size, len(captions))
        all_results.extend(
            _compute_loss_batch_core_with_hidden(
                model,
                processor,
                _safe_stack(hs_list[start:end]),
                images[start:end],
                captions[start:end],
            )
        )
    return all_results


# ─────────────────────────────────────────────
#  FEATURE 1: Loss Ratio
#  score = loss_base / loss_finetuned
#  Intuition: fine-tuned model memorises seen pairs → lower loss than base
# ─────────────────────────────────────────────


def feature_loss_ratio(
    finetuned_model,
    finetuned_proc,
    base_model,
    base_proc,
    image: Image.Image,
    caption: str,
    loss_ft: float = None,
) -> float:
    if loss_ft is None:
        loss_ft = compute_loss(finetuned_model, finetuned_proc, image, caption)
    if base_model is None:
        return -loss_ft  # fallback: just use negative loss
    loss_base = compute_loss(base_model, base_proc, image, caption)
    # ratio > 1 means fine-tuned model is more confident → member signal
    return loss_base / (loss_ft + 1e-8)


# ─────────────────────────────────────────────
#  FEATURE 2: Caption Contrast (joint-pair signal)
#  score = mean(loss_shuffled) - loss_original
#  Intuition: for seen pairs the model aligns this specific (img, cap)
#             much better than random other captions
# ─────────────────────────────────────────────


def feature_caption_contrast(
    model,
    processor,
    image: Image.Image,
    caption: str,
    all_captions: list,  # pool to sample negatives from
    n_shuffles: int = 5,
    loss_original: float = None,
    batch_size: int = 8,
) -> float:
    if loss_original is None:
        loss_original = compute_loss(model, processor, image, caption)

    # sample n_shuffles captions that are NOT the original
    negatives = [c for c in all_captions if c != caption]
    sampled = random.sample(negatives, min(n_shuffles, len(negatives)))

    # batch compute losses for shuffled captions (same image, different captions)
    images_batch = [image] * len(sampled)
    neg_losses = compute_loss_batch(model, processor, images_batch, sampled)
    loss_neg = np.mean(neg_losses)

    # large positive gap → this specific caption fits the image unusually well
    return float(loss_neg - loss_original)


# ─────────────────────────────────────────────
#  FEATURE 3: Image Corruption Sensitivity
#  score = -(mean(loss_corrupted) - loss_clean)
#  Intuition: model is "robust" to small perturbations of seen images
#             because it has memorised them; loss barely rises when corrupted
# ─────────────────────────────────────────────


def corrupt_image(
    image: Image.Image, noise_std: float, blur_radius: float
) -> Image.Image:
    """Apply Gaussian blur + additive Gaussian noise."""
    img = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr + np.random.normal(0, noise_std, arr.shape)
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def feature_corruption_sensitivity(
    model,
    processor,
    image: Image.Image,
    caption: str,
    n_corruptions: int = 3,
    noise_std: float = 0.05,
    blur_radius: float = 2.0,
    loss_clean: float = None,
) -> float:
    if loss_clean is None:
        loss_clean = compute_loss(model, processor, image, caption)

    # batch compute losses for corrupted images (different images, same caption)
    corrupted_images = [
        corrupt_image(image, noise_std, blur_radius) for _ in range(n_corruptions)
    ]
    captions_batch = [caption] * n_corruptions
    corrupted_losses = compute_loss_batch(
        model, processor, corrupted_images, captions_batch
    )
    loss_corrupted = np.mean(corrupted_losses)

    # return NEGATIVE delta so larger = more robust = more likely member
    return float(-(loss_corrupted - loss_clean))


# ─────────────────────────────────────────────
#  MAIN FEATURE EXTRACTION
# ─────────────────────────────────────────────

FEATURE_NAMES = ["loss_ratio", "caption_contrast", "corruption_sensitivity"]


def extract_all_features(
    dataset,
    finetuned_model,
    finetuned_proc,
    base_model,
    base_proc,
    all_captions: list,
    max_samples=None,
) -> pd.DataFrame:
    """
    Optimised feature extraction with three speedups:
      1. torch.compile  (applied in load_model)
      2. Image hidden-state caching — vision encoder runs ONCE per unique image;
         shuffled-caption variants reuse the cached hidden states.
      3. CPU prefetch pipeline — a background thread loads and decodes images
         while the GPU processes the previous batch.
    """
    cfg = CONFIG
    debug_max = cfg.get("debug_max")
    effective_max = debug_max if debug_max else max_samples

    records = []

    # ── Prefetch thread: overlaps image decoding with GPU compute ──────────
    dataset_iter = iter(dataset)
    prefetch_q: Queue = Queue(maxsize=3)

    def _prefetch_worker():
        seen = 0
        while effective_max is None or seen < effective_max:
            remaining = (effective_max - seen) if effective_max else cfg["batch_size"]
            batch = []
            for _ in range(min(cfg["batch_size"], remaining)):
                try:
                    batch.append(next(dataset_iter))
                except StopIteration:
                    break
            if not batch:
                break
            images_pre = [extract_image(ex) for ex in batch]
            captions_pre = [ex["text"] for ex in batch]
            prefetch_q.put((batch, images_pre, captions_pre))
            seen += len(batch)
        prefetch_q.put(None)  # sentinel

    worker = Thread(target=_prefetch_worker, daemon=True)
    worker.start()

    pbar = tqdm(total=effective_max, desc="Extracting features")

    while True:
        item = prefetch_q.get()
        if item is None:
            break

        batch_examples, images, captions = item
        B = len(batch_examples)

        # ── 1. Vision encoder: ONE pass per batch of images ────────────────
        ft_img_hs = extract_image_hidden_states_batch(
            finetuned_model, finetuned_proc, images
        )

        if base_model is not None:
            base_img_hs = extract_image_hidden_states_batch(
                base_model, base_proc, images
            )

        # ── 2. Original losses (LLM only, vision encoder skipped) ──────────
        loss_ft_originals = compute_loss_batch_with_hidden(
            finetuned_model,
            finetuned_proc,
            ft_img_hs,
            images,
            captions,
        )

        if base_model is not None:
            loss_base_originals = compute_loss_batch_with_hidden(
                base_model,
                base_proc,
                base_img_hs,
                images,
                captions,
            )

        # ── 3. Shuffled captions — reuse ft_img_hs, no extra vision pass ───
        flat_shuf_imgs, flat_shuf_hs, flat_shuf_caps = [], [], []
        shuf_counts = []
        for i in range(B):
            negs = [c for c in all_captions if c != captions[i]]
            sampled = random.sample(negs, min(cfg["n_shuffles"], len(negs)))
            n_s = len(sampled)
            flat_shuf_imgs.extend([images[i]] * n_s)
            flat_shuf_hs.extend([ft_img_hs[i]] * n_s)
            flat_shuf_caps.extend(sampled)
            shuf_counts.append(n_s)

        flat_shuf_losses = compute_loss_batch_with_hidden(
            finetuned_model,
            finetuned_proc,
            flat_shuf_hs,
            flat_shuf_imgs,
            flat_shuf_caps,
        )

        # ── 4. Corrupted images — new images, full forward pass required ───
        flat_corr_imgs, flat_corr_caps = [], []
        for i in range(B):
            for _ in range(cfg["n_corruptions"]):
                flat_corr_imgs.append(
                    corrupt_image(images[i], cfg["noise_std"], cfg["corruption_radius"])
                )
                flat_corr_caps.append(captions[i])

        flat_corr_losses = compute_loss_batch(
            finetuned_model, finetuned_proc, flat_corr_imgs, flat_corr_caps
        )

        # ── 5. Assemble features ────────────────────────────────────────────
        idx_shuf, idx_corr = 0, 0
        for i, ex in enumerate(batch_examples):
            loss_ft = loss_ft_originals[i]

            f1 = (loss_base_originals[i] / (loss_ft + 1e-8)) if base_model else -loss_ft

            n_s = shuf_counts[i]
            shuf_l = flat_shuf_losses[idx_shuf : idx_shuf + n_s]
            f2 = float(np.mean(shuf_l) - loss_ft) if shuf_l else 0.0
            idx_shuf += n_s

            n_c = cfg["n_corruptions"]
            corr_l = flat_corr_losses[idx_corr : idx_corr + n_c]
            f3 = float(-(np.mean(corr_l) - loss_ft)) if corr_l else 0.0
            idx_corr += n_c

            records.append(
                {
                    "id": ex["id"],
                    "is_member": ex.get("is_member", 0),
                    "loss_ratio": f1,
                    "caption_contrast": f2,
                    "corruption_sensitivity": f3,
                }
            )

        pbar.update(B)

    pbar.close()
    worker.join(timeout=5)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  EVALUATION HELPERS
# ─────────────────────────────────────────────


def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.1) -> float:
    """Compute TPR at a given FPR threshold (for TPR@FPR=0.1 metric)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def evaluate(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    tpr = tpr_at_fpr(y_true, y_score, target_fpr=0.1)
    print(f"  AUC              : {auc:.4f}")
    print(f"  TPR @ FPR=0.1    : {tpr:.4f}")
    return auc, tpr


# ─────────────────────────────────────────────
#  TRAIN & PREDICT PIPELINE
# ─────────────────────────────────────────────


def train_and_predict(recompute=False):
    cfg = CONFIG
    cache_dir = cfg["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    # ── Check cache ────────────────────────────
    splits = {
        "train": ("train", "Train"),
        "val": ("validation", "Val"),
        "test": ("test", "Test"),
    }

    mode_suffix = "_with_base" if cfg["base_model_path"] else "_no_base"

    dfs = {}
    need_extraction = False
    for split_key, (hf_split, desc) in splits.items():
        cache_path = os.path.join(cache_dir, f"{split_key}{mode_suffix}.json")
        if os.path.exists(cache_path) and not recompute:
            print(f"Loading cached {split_key} features from {cache_path}")
            dfs[split_key] = pd.read_json(cache_path)
        else:
            need_extraction = True

    if need_extraction:
        # ── Build caption pool (shuffled to avoid distribution bias) ──
        print("Building caption pool...")
        all_captions = []
        for attempt in range(5):
            try:
                ds_pool = load_dataset(
                    cfg["dataset_id"], split="train", streaming=True
                ).shuffle(seed=cfg["seed"], buffer_size=10000)
                for ex in ds_pool:
                    all_captions.append(ex["text"])
                    if len(all_captions) >= 6000:
                        break
                break  # success
            except Exception as e:
                print(f"  Attempt {attempt+1}/5 failed: {e}")
                all_captions.clear()
                import time

                time.sleep(5 * (attempt + 1))
        if not all_captions:
            raise RuntimeError("Failed to build caption pool after 5 attempts")

        # ── Load models ────────────────────────
        print("Loading fine-tuned model...")
        ft_model, ft_proc = load_model(cfg["finetuned_model_path"])

        base_model, base_proc = None, None
        if cfg["base_model_path"]:
            print("Loading base model...")
            base_model, base_proc = load_model(cfg["base_model_path"])

        # ── Extract & cache each split ─────────
        for split_key, (hf_split, desc) in splits.items():
            cache_path = os.path.join(cache_dir, f"{split_key}{mode_suffix}.json")
            if split_key in dfs:
                continue  # already loaded from cache

            print(f"\n=== Extracting {desc} features ===")
            ds = load_dataset(cfg["dataset_id"], split=hf_split, streaming=True)
            df = extract_all_features(
                ds, ft_model, ft_proc, base_model, base_proc, all_captions
            )
            df.to_json(cache_path, orient="records", indent=2)
            print(f"  -> cached to {cache_path}")
            dfs[split_key] = df

        # Free model memory
        del ft_model, ft_proc, base_model, base_proc
        if cfg["device"] == "cuda":
            torch.cuda.empty_cache()

    df_train = dfs["train"]
    df_val = dfs["val"]
    df_test = dfs["test"]

    # ── Normalise ──────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_NAMES])
    y_train = df_train["is_member"].values
    X_val = scaler.transform(df_val[FEATURE_NAMES])
    y_val = df_val["is_member"].values
    X_test = scaler.transform(df_test[FEATURE_NAMES])

    # ── Train logistic regression fusion ───────
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg["seed"])
    clf.fit(X_train, y_train)

    # ── Validate ────────────────────────────────
    print("\n── Validation results ──")
    val_scores = clf.predict_proba(X_val)[:, 1]
    evaluate(y_val, val_scores)

    # Sanity check: per-feature AUC on val set
    print("\n── Per-feature AUC on val ──")
    for i, name in enumerate(FEATURE_NAMES):
        auc = roc_auc_score(y_val, X_val[:, i])
        print(f"  {name:30s}: {auc:.4f}")

    # ── Generate submission ─────────────────────
    test_scores = clf.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(
        {
            "id": df_test["id"],
            "score": test_scores,
        }
    )
    project_dir = "/content/drive/MyDrive/VLM_MIA_Project"
    os.makedirs(project_dir, exist_ok=True)

    submission_path = os.path.join(project_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print(submission.head())

    return submission


# ─────────────────────────────────────────────
#  ABLATION: if you don't have a base model,
#  use this lightweight single-feature fallback
# ─────────────────────────────────────────────


def predict_loss_only(dataset, model, processor, max_samples=None) -> np.ndarray:
    """
    Simplest possible baseline: just rank by -loss.
    Useful as a sanity check or when no base model is available.
    """
    scores = []
    for i, example in enumerate(tqdm(dataset, total=max_samples)):
        if max_samples and i >= max_samples:
            break
        image = extract_image(example)
        caption = example["text"]
        loss = compute_loss(model, processor, image, caption)
        scores.append(-loss)  # lower loss → higher membership score
    return np.array(scores)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 手动控制是否重新计算
    RECOMPUTE = True
    train_and_predict(recompute=RECOMPUTE)
