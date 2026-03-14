# Milestone 1 Progress Report

**Team:** TFC
**Members:** Tianhao Cao, Yusen Huang, Marco Wang, Darwin Zhang

------------------------------------------------------------------------

## Model 1: Base Model Likelihood Ratio (Reference Model)

### Model Description
In our baseline approach (Milestone 1), we used the sequence-level cross-entropy loss of the target fine-tuned model (`UBC-SLIME/colx_531_smollm2-135m`) as the membership proxy. However, raw loss is flawed because some clinical notes are inherently easier or harder to predict regardless of their membership status, leading to false classifications based on linguistic complexity.

To address this, we implemented a **Base Model Likelihood Ratio (Reference Model)** approach. This method compares the loss of the target fine-tuned model against a reference foundation model (`HuggingFaceTB/SmolLM2-135M`) that has not seen the fine-tuning data. The final membership score is computed as the difference in loss: `loss_base - loss_target`. If a sample was in the fine-tuning dataset, the fine-tuned model evaluates it with a significantly lower loss than the base model, isolating the "memorization" signal from the text's inherent complexity.

### Implementation
We implemented this method in `milestone2/src/reference_model.py`. The script loads both the target model and the base model, tokenizes the input text, and runs inference on both models sequentially. The final score is computed as `loss_base - loss_target`, and the results are saved to a CSV file.

### Results
Because Kaggle test set labels are hidden, we performed local evaluation using the `validation` split to assess the performance of our new method against the baseline. 

| Method | AUC | TPR@FPR=0.1 |
| :--- | :---: | :---: |
| Baseline (Raw Loss) | 0.5839 | 0.1552 |
| Reference Model (Lkhood Ratio) | 0.6701 | 0.2214 |

### Discussions
The Base Model Likelihood Ratio successfully isolated the memorization signal from the inherent linguistic complexity of the medical texts. By subtracting the foundation model's loss from the finetuned model's loss, we removed the bias where naturally difficult texts were falsely classified as non-members. As evaluated on the validation set, this normalization resulted in a substantial 43% relative improvement in True Positive Rate at a 10% False Positive Rate (increasing from 15.5% to 22.1%), and increased the overall AUC from 0.58 to 0.67.



## Contributions

### Milestone 2

| Member       | Contributions         | Percentage |
|--------------|-----------------------|------------|
| Tianhao Cao  | Background Research   | 25%        |
| Yusen Huang  | Metrics and Baseline  | 25%        |
| Marco Wang   | Task and Introduction | 25%        |
| Darwin Zhang | Data and Baseline     | 25%        |

**Total:** 100%
