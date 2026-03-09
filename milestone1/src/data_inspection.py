from datasets import load_dataset
import pandas as pd
import numpy as np

def main():
    print("Loading dataset 'UBC-SLIME/colx_531_group_project'...")
    dataset = load_dataset("UBC-SLIME/colx_531_group_project")
    
    # Print dataset dictionary strcuture 
    print("\nDataset Structure:")
    print(dataset)

    # Get samples to view
    print("\n" + "="*50)
    print("SAMPLES FOR data_inspection.md")
    print("="*50)
    
    # First 3 samples 
    if 'train' in dataset:
        train_data = dataset['train']
        for i in range(3):
            print(f"\n--- Sample {i+1} ---")
            print(train_data[i])
    
    # Descriptive statistics 
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS FOR PROGRESS REPORT")
    print("="*50)
    
    stats = []
    
    def get_length(example):
        # Calculating length in terms of character length of dictionary string 
        return len(str(example))

    for split in dataset.keys():
        split_data = dataset[split]
        num_samples = len(split_data)
        
        if num_samples > 0:
            lengths = [get_length(ex) for ex in split_data]
            mean_len = np.mean(lengths)
            max_len = np.max(lengths)
            min_len = np.min(lengths)
        else:
            mean_len = max_len = min_len = 0
            
        stats.append({
            "Split": split,
            "Number of samples": num_samples,
            "Mean Length": round(mean_len, 2),
            "Max length": max_len,
            "Min length": min_len
        })
        
    df_stats = pd.DataFrame(stats)
    print("\n" + df_stats.to_markdown(index=False))

if __name__ == "__main__":
    main()