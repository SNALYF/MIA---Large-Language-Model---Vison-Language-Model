import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    # Using mps for local, switch for Colab
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Loading model and tokenizer 
    model_id = "UBC-SLIME/colx_531_smollm2-135m"
    print(f"Loading model and tokenizer: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Not required to pad by token, so set to eos 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    # Load test set 
    print("Loading test dataset...")
    dataset = load_dataset("UBC-SLIME/colx_531_group_project", split="test")
    
    predictions = []

    # Running inference 
    print("Running baseline inference...")

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
            
            # Passing labels=input_ids makes the model automatically calculate cross-entropy loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # Get the loss as a standard python float
            loss = outputs.loss.item()
            
            # We want a higher score for members. 
            # Lower loss = more likely member. Therefore, score = -loss
            score = -loss 
            
            predictions.append({"id": doc_id, "score": score})

    # Saving to csv 
    output_file = "milestone1/baseline_submission.csv"
    df_preds = pd.DataFrame(predictions)
    df_preds.to_csv(output_file, index=False)
    print(f"\Predictions saved to {output_file}")

if __name__ == "__main__":
    main()