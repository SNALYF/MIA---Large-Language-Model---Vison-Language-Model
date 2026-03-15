# Milestone 2 Group Report

**Team:** TFC **Members:** Tianhao Cao, Yusen Huang, Marco Wang, Darwin Zhang

------------------------------------------------------------------------

## Model 1: Base Model Likelihood Ratio (Reference Model)

### Model Description

In our baseline approach (Milestone 1), we used the sequence-level cross-entropy loss of the target fine-tuned model (`UBC-SLIME/colx_531_smollm2-135m`) as the membership proxy. However, raw loss is flawed because some clinical notes are inherently easier or harder to predict regardless of their membership status, leading to false classifications based on linguistic complexity.

To address this, we implemented a **Base Model Likelihood Ratio (Reference Model)** approach. This method compares the loss of the target fine-tuned model against a reference foundation model (`HuggingFaceTB/SmolLM2-135M`) that has not seen the fine-tuning data. The final membership score is computed as the difference in loss: `loss_base - loss_target`. If a sample was in the fine-tuning dataset, the fine-tuned model evaluates it with a significantly lower loss than the base model, isolating the "memorization" signal from the text's inherent complexity.

### Implementation

We implemented this method in `milestone2/src/reference_model.py`. The script loads both the target model and the base model, tokenizes the input text, and runs inference on both models sequentially. The final score is computed as `loss_base - loss_target`, and the results are saved to a CSV file.

### Results

Because Kaggle test set labels are hidden, we performed local evaluation using the `validation` split to assess the performance of our new method against the baseline.

| Method | AUC | [TPR\@FPR](mailto:TPR@FPR){.email}=0.1 |
|:---|:--:|:--:|
| Baseline (Raw Loss) | 0.5839 | 0.1552 |
| Reference Model (Lkhood Ratio) | 0.6701 | 0.2214 |

### Discussions

The Base Model Likelihood Ratio successfully isolated the memorization signal from the inherent linguistic complexity of the medical texts. By subtracting the foundation model's loss from the finetuned model's loss, we removed the bias where naturally difficult texts were falsely classified as non-members. As evaluated on the validation set, this normalization resulted in a substantial 43% relative improvement in True Positive Rate at a 10% False Positive Rate (increasing from 15.5% to 22.1%), and increased the overall AUC from 0.58 to 0.67.

## Model 2: **Min-K% Prob**

### Model Description

**Min-K% Prob** (Shi et al., 2024) is Instead of using average loss, it looks at the k% of tokens with the lowest log-probabilities. For a given text, we compute the log-probability the model assigns to each token, then select only the bottom *k%* of tokens — the ones the model found most "surprising." The final score is the mean log-probability of this subset. The intuition is that when a model has been fine-tuned on a particular text, even the tokens it finds *least* predictable should still receive relatively high probability compared to unseen text.

### Implementation

We implemented this method in `milestone2/src/min_k_prob.py`.

### Results

Because Kaggle test set labels are hidden, we performed local evaluation using the `validation` split to assess the performance of our new method against the baseline.

| Method              |  AUC   | [TPR\@FPR](mailto:TPR@FPR){.email}=0.1 |
|:--------------------|:------:|:--------------------------------------:|
| Baseline (Raw Loss) | 0.5839 |                 0.1552                 |
| Min-K% Prob         | 0.6166 |                 0.1682                 |

### Discussions

Min-K% Prob operates on the principle that memorization is most visible in the tails of the token-level probability distribution. A model that has memorized a text will assign non-trivially high probability even to tokens that are contextually difficult to predict (rare words, domain-specific terms, unusual phrasing). In contrast, for non-member text, these difficult tokens remain genuinely surprising, producing very low probabilities. As evaluated on the validation set, this normalization resulted in a minor 8.4% relative improvement in True Positive Rate at a 10% False Positive Rate (increasing from 15.5% to 16.8%), and increased the overall AUC from 0.58 to 0.61.

------------------------------------------------------------------------

### Literature Review

**Paper 1: A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models**

This paper addresses the task of statistically evaluating Membership Inference Attack (MIA) methods across thousands of experimental settings to resolve performance inconsistencies reported in previous literature. The researchers utilized the WikiMIA benchmark alongside data sampled from the Pile and Dolma pre-training corpora, covering diverse domains like arXiv, GitHub, and Wikipedia. The evaluation was conducted using the Pythia model suite (ranging from 160m to 12b parameters) and the OLMo series (1b to 13b parameters). The main contribution of this work is a large-scale statistical analysis revealing that while MIA performance generally improves with model size and varies by domain, most existing methods do not statistically outperform simple baselines. Additionally, the authors identified that deciding on an effective membership threshold is a significant real-world challenge and demonstrated that the final embedding layer typically used for MIA is suboptimal due to low separability.

**Short summary:**

Main Task: The paper statistically revisits Membership Inference Attack (MIA) methods across a wide variety of settings (thousands of experiments) to address the performance inconsistencies reported in previous studies.

Dataset: The researchers used the WikiMIA benchmark and sampled data from the Pile and Dolma pre-train corpora, covering domains such as arXiv, GitHub, FreeLaw, and Wikipedia.

Model: The study evaluated the Pythia model series (160m to 12b parameters) and the OLMo series (1b to 13b parameters).

Methods Used: - Probabilistic Attacks: Profiling standard loss-based (log-likelihood) and perplexity methods to establish baseline performance. - Statistical Thresholding: Investigating the difficulty of setting a universal decision threshold for "member" vs. "non-member" status across different model sizes. - Embedding Analysis: Testing the separability of data within the final embedding layer, which was found to be suboptimal for MIA. - Comparative Baselines: Evaluating advanced methods like Neighborhood attacks and Min-K% Prob within their statistical framework.

Main Contribution: The authors conducted a large-scale statistical analysis showing that while MIA performance generally scales with model size, most methods do not statistically outperform simple baselines. They also identified that finding a unified threshold is a major challenge due to its variance across domains and model sizes, and demonstrated that the final embedding layer used by many current methods is suboptimal for MIA tasks.

*Citation: Chen, B., Han, N., & Miyao, Y. (2025). A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics.*

------------------------------------------------------------------------

**Paper 2: RECALL: Membership Inference via Relative Conditional Log-Likelihoods**

The main task of this paper is to detect pretraining data in large language models by leveraging their conditional language modeling capabilities. The authors conducted extensive experiments using the WikiMIA and MIMIR (derived from the Pile) benchmarks. A wide array of models were tested, including the Pythia family (160M to 12B), GPT-NeoX 20B, LLaMA (13B and 30B), OPT 66B, and the state-space model Mamba 1.4B. The main contribution is the introduction of RECALL (Relative Conditional Log-Likelihood), a novel MIA that detects membership by measuring the relative change in log-likelihood when target data is prefixed with non-member context. This method achieves state-of-the-art results on WikiMIA and demonstrates that synthetic prefixes can be as effective as real non-member data, significantly enhancing the practical utility of membership audits.

**Short summary:**

Main Task: This paper proposes a novel Membership Inference Attack called RECALL to detect if specific data was used in an LLM's pretraining by leveraging the model's conditional language modeling capabilities.

Dataset: The method was evaluated on the WikiMIA and MIMIR benchmarks, the latter of which includes various domains like Wikipedia, GitHub, and PubMed Central.

Model: Experiments were conducted on diverse architectures including Pythia (160M to 12B), GPT-NeoX 20B, LLaMA (13B and 30B), OPT 66B, and the Mamba 1.4B state-space model.

Methods Used: - RECALL Score: Implementing a ratio of conditional log-likelihood (where text is prefixed with non-member context) to unconditional log-likelihood. - Non-member Prefixing: Using synthetic, random, or real-world non-member text as a "reference context" to observe the drop in the model's prediction certainty. - Ensemble Strategy: Combining multiple prefixes to stabilize the membership signal and improve detection accuracy.

Main Contribution: The paper introduces the RECALL score, which effectively identifies members by the larger decrease in likelihood they experience when conditioned on unseen text. The method achieved state-of-the-art performance on WikiMIA and proved robust even with random prefixes, significantly enhancing the practical utility of membership audits.

*Citation: Xie, R., Wang, J., Huang, R., Zhang, M., Dhingra, B., & Ge, R. (2024). RECALL: Membership Inference via Relative Conditional Log-Likelihoods. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.*

------------------------------------------------------------------------

## Contributions

### Milestone 2

| Member       | Contributions     | Percentage |
|--------------|-------------------|------------|
| Tianhao Cao  | Methods           | 25%        |
| Yusen Huang  | Methods           | 25%        |
| Marco Wang   | Methods           | 25%        |
| Darwin Zhang | Literature Review | 25%        |

**Total:** 100%
