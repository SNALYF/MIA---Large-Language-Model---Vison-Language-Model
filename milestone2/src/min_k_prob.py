import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

def compute_min_k_percent_score(logits, input_ids, k_percent=20):
    """
    Compute the Min-K% Prob score for a sequence.
    
    Intuition: Training members have fewer "surprising" tokens,
    so even their worst-scoring tokens still have relatively high probability.
    Higher score (less negative) => more likely a member.
    """
    # Shift so that logits[t] predicts input_ids[t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute log-softmax over vocabulary dimension
    log_probs = F.log_softmax(shift_logits, dim=-1)  

    # Gather the log-prob of the actual next token at each position
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  

    token_log_probs = token_log_probs.squeeze(0)  

    seq_len = token_log_probs.size(0)
    if seq_len == 0:
        return 0.0

    # Select the bottom k% of token log-probs
    k = max(1, int(seq_len * k_percent / 100))
    bottom_k, _ = torch.topk(token_log_probs, k=k, largest=False)

    # Score = mean of the k lowest log-probs
    score = bottom_k.mean().item()
    return score


def main():
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Load model and tokenizer (finetuned target model)
    model_id = "UBC-SLIME/colx_531_smollm2-135m"
    print(f"Loading model and tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    # Load test set
    print("Loading test dataset...")
    dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="test")

    predictions = []
    max_length = 1024
    k_percent = 20  # Hyperparameter: percentage of lowest-prob tokens to use

    print(f"Running Min-K% Prob inference (k={k_percent}%)...")

    with torch.no_grad():
        for example in tqdm(dataset, desc="Scoring Test Set"):
            doc_id = example["id"]
            text = example["text"]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            score = compute_min_k_percent_score(
                logits, inputs["input_ids"], k_percent=k_percent
            )

            predictions.append({"id": doc_id, "score": score})

    # Save predictions
    os.makedirs("milestone2", exist_ok=True)
    output_file = "milestone2/min_k_prob_submission.csv"
    df_preds = pd.DataFrame(predictions)
    df_preds.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
