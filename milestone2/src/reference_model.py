import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

def main():
    # Using mps for local, switch for Colab
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Loading target model (finetuned)
    target_model_id = "UBC-SLIME/colx_531_smollm2-135m"
    print(f"Loading target model and tokenizer: {target_model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    # Not required to pad by token, so set to eos 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    target_model = AutoModelForCausalLM.from_pretrained(target_model_id).to(device)
    target_model.eval()

    # Loading base model (reference)
    base_model_id = "HuggingFaceTB/SmolLM2-135M"
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to(device)
    base_model.eval()

    # Load test set 
    print("Loading test dataset...")
    dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="test")
    
    predictions = []

    # Running inference 
    print("Running Base Model Likelihood Ratio inference...")

    # As we are running local, we halved the max_length 
    max_length = 1024 

    with torch.no_grad():
        for example in tqdm(dataset, desc="Scoring Test Set"):
            doc_id = example['id']
            text = example['text']
            
            # Tokenize and move to device
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            ).to(device)
            
            # Get loss from target model
            target_outputs = target_model(**inputs, labels=inputs["input_ids"])
            loss_target = target_outputs.loss.item()
            
            # Get loss from base model
            base_outputs = base_model(**inputs, labels=inputs["input_ids"])
            loss_base = base_outputs.loss.item()
            
            # Score: loss_base - loss_target
            # Higher score = more likely to be a member
            # (Finetuned model has much lower loss than base model)
            score = loss_base - loss_target
            
            predictions.append({"id": doc_id, "score": score})

    # Saving to csv 
    os.makedirs("milestone2", exist_ok=True)
    output_file = "milestone2/reference_model_submission.csv"
    df_preds = pd.DataFrame(predictions)
    df_preds.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
