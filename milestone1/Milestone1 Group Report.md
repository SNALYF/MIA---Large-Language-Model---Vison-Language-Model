# Milestone 1 Progress Report

**Team:** TFC  
**Members:** Tianhao Cao, Yusen Huang, Marco Wang, Darwin Zhang

---

## Task Description

<!-- 
Answer the following questions:
- What is the task?
- Why is the task important?
- What does this task try to achieve?
-->

/section{Task Description}
The task in this project is to perform a membership inference attack (MIA) on finetuned language models. More specifically, we aim to predict whether a given sample was included in the models’ finetuning dataset by assigning each sample a membership score. This task is important because it helps evaluate whether a model leaks information about its training data, which is closely related to privacy and memorization risks in machine learning systems. Therefore, the goal of this project is to develop a method that can distinguish member samples from non-member samples as accurately as possible and submit these predictions to the Kaggle competition.

## Data

<!-- 
Provide:
- A high-level verbal description of your data
- Descriptive statistics of the official dataset (table below)
-->

| Split | Number of Samples | Mean Length | Max Length | Min Length |
|-------|-------------------|-------------|------------|------------|
| Train |                   |             |            |            |
| Dev   |                   |             |            |            |
| Test  |                   |             |            |            |

## Evaluation Metrics

<!-- 
For each official evaluation metric, answer:
- What is the definition of this metric?
- What does this evaluation metric measure?
- How is this metric calculated?
-->

## Background

<!-- 
Summarize n-1 = 3 research papers from the references of the official task page.
For each paper, answer:
- What is the main task this paper tries to solve?
- What is the dataset used in this paper?
- What is the model used in this paper?
- What is the main contribution of this paper?

Remember to cite these papers properly using ACL citation format.
-->
\section{Background}
Membership Inference Attacks (MIAs), as an approach to identify whether a specific data point belongs to a target model's training dataset, have been widely utilized, primarily in traditional machine learning models \citep{shokri2017membership}. Although some studies \citep{shi2023detecting, meeus2023did} indicate that MIAs are effective on LLMs, recent research disproves this argument, demonstrating that MIAs barely outperform random guessing when evaluating the pre-training data of large language models \citep{duan2024do}.

\citet{duan2024do} conducted a performance analysis of MIAs on various models ranging from 70 million to 12 billion parameters. The evaluation targeted the following models:
\begin{itemize}
    \item Pythia: 70M, 160M, 1.4B, 2.8B, 6.9B, 12B 
    \item GPT-Neo: 125M, 1.3B, 2.7B
    \item Datablations: 2.8B
    \item SILO: 1.3B
    \item OLMo: 1B, 7B
\end{itemize}
The evaluation was performed on training sets composed of Wikipedia, PubMed Central, ArXiv, Pile-CC, Github, DM Math, and HackerNews. The research concludes that MIA's unreliable performance in evaluating data leakage is primarily due to two factors: first, the large scale of training data combined with near-one-epoch training limits model overfitting; second, there is an inherently fuzzy boundary, or high n-gram overlap, between member data and non-member data.

Furthermore, \citet{mattern2023membership} proposed an alternative methodology to bypass the reliance on reference datasets by conducting a Neighbourhood Attack. This attack was evaluated during the fine-tuning phase on the 117M parameter version of GPT-2. The target model was fine-tuned on 60,000 samples from the AG News corpus, 150,000 samples from Twitter, and 100,000 samples from Wikitext-103, while the neighbour texts were dynamically generated using a pre-trained BERT model. The paper finds that the Neighbourhood attack can outperform traditional reference-based MIAs (e.g., LiRA) by up to 100\% under realistic assumptions.

Moreover, \citet{fu2024membership} advance this line of research by introducing SPV-MIA (Self-calibrated Probabilistic Variation MIA), an architecture composed of two main modules: a Self-prompt Reference Model and a Probabilistic Variation Assessment[cite: 19, 80]. The authors evaluated this method on four target models (GPT-2, GPT-J, Falcon-7B, and LLaMA-7B) fine-tuned on the Wikitext-103, AG News, and XSum datasets[cite: 268]. The results demonstrate that SPV-MIA is highly robust and effectively bypasses the unrealistic reference data assumptions of prior reference-based attacks[cite: 83, 92]. Specifically, it significantly elevates attack performance, increasing the AUC from approximately 0.7 to an average of 0.924 [cite: 23, 287], while achieving an average True Positive Rate (TPR) of 46.9\% at a 1\% False Positive Rate (FPR).

## Week 1 Baseline

<!-- 
Describe your baseline method:
- How did you build your baseline?
- What is the motivation behind this baseline?
- What hardware (CPUs, GPUs, memory, etc.) did you use?
- How did it work?
-->

## Contributions

### Milestone 1

<!-- 
Each team member must write down their individual contributions.
Specify the percentage of contributions (must sum to 1).
-->

| Member        | Contributions | Percentage |
|---------------|---------------|------------|
| Tianhao Cao   |               |            |
| Yusen Huang   |               |            |
| Marco Wang    |               |            |
| Darwin Zhang  |               |            |

**Total:** 100%
