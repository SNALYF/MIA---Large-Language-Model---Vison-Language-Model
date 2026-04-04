# Milestone 5 Group Report

**Team:** TFC **Members:** Tianhao Cao, Yusen Huang, Marco Wang, Darwin Zhang

## Literature Review

**Paper 1: M⁴I: Multi-modal Models Membership Inference**

This paper addresses the privacy risks of multi-modal machine learning models by proposing the first membership inference attack (MIA) framework specifically designed for models that process multiple modalities. Prior MIA research had focused almost exclusively on single-modal models such as image classifiers or tabular data models, leaving the rapidly growing field of vision-language models largely unexplored. The authors argue that the cross-modal nature of these systems — where an image input produces a text output, for example — introduces unique vulnerabilities that require new attack strategies beyond simply probing a single output distribution.

**Short summary:**

**Main Task**: Determining whether a specific data record (such as an image-caption pair) was used to train a multi-modal model. This is formulated as a binary classification problem: given a target sample and query access to the model, classify it as a member (seen during training) or a non-member.

**Dataset**: The authors evaluate their attacks on three image captioning datasets — **MS-COCO**, **FLICKR8k**, and **IAPR TC-12** — covering general-domain image-caption pairs. They also apply the attack to a **medical report generation** model, demonstrating the framework's applicability across domains.

**Models**: The paper proposes two attack methods — **Metric-based M⁴I (MB-M⁴I)**, which uses text generation similarity metrics (**ROUGE** and **BLEU**) between the model's output and the target caption to infer membership; and **Feature-based M⁴I (FB-M⁴I)**, which employs a pre-trained shadow multi-modal feature extractor and uses **Euclidean distance** in the shared feature space to compare input-output similarity. Target captioning models follow the **Show-and-Tell** setting — encoder-decoder architectures with a **ResNet-152** or **VGG-16** image encoder and an **LSTM** text decoder.

**Main Contribution**: M⁴I presents itself as the first work to propose an MIA framework explicitly targeting multi-modal models. The metric-based attack achieves an average success rate of **72.5%**, while the feature-based attack achieves **94.83%** in unrestricted scenarios. These results demonstrate that membership inference attacks are effective against multi-modal models and establish a foundational baseline for subsequent VLM privacy research.

---

**Paper 2: OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of Membership Inference Attacks on Large Vision-Language Models**

This paper critically examines the reliability of existing MIA research on large vision-language models (LVLMs), raising an important methodological concern: many high attack success rates reported in prior work may be artifacts of **distributional bias** in dataset construction rather than evidence of genuine memorization detection. When member and non-member samples come from different distributions (e.g., different time periods, sources, or collection methods), an attack may simply learn to distinguish those distributions rather than identify true training membership. To address this, the authors introduce a carefully controlled benchmark designed to remove such biases.

**Short summary:**

**Main Task**: Benchmarking the effectiveness of state-of-the-art MIA methods on LVLMs under fair, unbiased evaluation conditions. The paper asks whether existing MIA methods can truly detect training membership, or whether prior high success rates are inflated by distributional shortcuts.

**Dataset**: The authors introduce **OpenLVLM-MIA**, a controlled benchmark of **6,000 images** (1,000 members and 1,000 non-members per training stage), with ground-truth membership labels across three distinct LVLM training stages: (1) vision encoder pre-training, (2) projector pre-training, and (3) instruction tuning. The member and non-member distributions are carefully balanced to remove any exploitable data bias.

**Models**: The benchmark is built on an **OpenCLIP + LLaVA** architecture with a fully transparent, open-source training pipeline. The evaluation uses a **gray-box setting**, where the attacker has access to the model's output logits and generated text (but not internal weights). Evaluated MIA methods include **Perplexity**, **Min-K% Probability**, and **Max Rényi** variants.

**Main Contribution**: The paper demonstrates that when distributional bias is eliminated, SOTA MIA methods converge to **near-random-chance performance** on LVLMs. This suggests that much of the progress reported in prior VLM-MIA literature reflects dataset construction artifacts rather than genuine privacy vulnerabilities. OpenLVLM-MIA provides a rigorous and transparent foundation for future MIA research on LVLMs, and directly motivates the need for more sophisticated attack strategies beyond simple likelihood-based scoring.

---

*Citation: Hu, P., Wang, Z., Sun, R., Wang, H., & Xue, M. (2022). M⁴I: Multi-modal Models Membership Inference. Advances in Neural Information Processing Systems, 35 (NeurIPS 2022). arXiv:2209.06997.*

*Citation: Miyamoto, R., Fan, X., Kido, F., Matsumoto, T., & Yamana, H. (2025). OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of Membership Inference Attacks on Large Vision-Language Models. arXiv:2510.16295.*

---
## Contributions

### Milestone 4
| Member       | Contributions     | Percentage |
|--------------|-------------------|------------|
| Tianhao Cao  | Methods           | 25%        |
| Yusen Huang  | Methods           | 25%        |
| Marco Wang   | Methods           | 25%        |
| Darwin Zhang | Literature Review | 25%        |

**Total:** 100%