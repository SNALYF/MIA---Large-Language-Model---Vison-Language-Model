from datasets import load_dataset
import pandas as pd
import numpy as np

def main():
    dataset = load_dataset("UBC-SLIME/colx585_group_project_data")
    
    print("\nDataset Structure:")
    print(dataset)

    # Extracting samples for data inspection, 3 samples 
    split_to_sample = 'train' if 'train' in dataset else list(dataset.keys())[0]
    sample_data = dataset[split_to_sample]
    
    for i in range(3):
        example = sample_data[i]
        
        # Case: if image data crashes terminal with its outputs
        printable_example = {}
        for key, value in example.items():
            # Check if the value is an image (usually has mode and size attributes)
            if hasattr(value, 'mode') and hasattr(value, 'size'):
                printable_example[key] = f"<Image object: mode={value.mode}, size={value.size}>"
            else:
                printable_example[key] = value
                
        print(printable_example)
    
    # Descriptive statistics 
    stats = []
    
    # Helper function to get text length for a VLM dataset
    def get_text_length(example):
        # Combines all text fields in the sample to calculate length
        text_content = ""
        for key, value in example.items():
            if isinstance(value, str):
                text_content += value + " "
        return len(text_content.strip()) if text_content else 0

    for split in dataset.keys():
        split_data = dataset[split]
        num_samples = len(split_data)
        
        if num_samples > 0:
            lengths = [get_text_length(ex) for ex in split_data]
            mean_len = np.mean(lengths)
            median_len = np.median(lengths)
            std_len = np.std(lengths)
            max_len = np.max(lengths)
            min_len = np.min(lengths)
        else:
            mean_len = median_len = std_len = max_len = min_len = 0
            
        stats.append({
            "Split": split,
            "Number of samples": num_samples,
            "Mean Text Length": round(mean_len, 2),
            "Median Text Length": round(median_len, 2),
            "Max length": max_len,
            "Min length": min_len,
            "Length Std Dev": round(std_len, 2)
        })
        
    df_stats = pd.DataFrame(stats)
    print("\n" + df_stats.to_markdown(index=False))

if __name__ == "__main__":
    main()