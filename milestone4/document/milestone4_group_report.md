# Milestone 4 Group Report

**Team:** TFC **Members:** Tianhao Cao, Yusen Huang, Marco Wang, Darwin Zhang

## Data

### High-Level Description

The `UBC-SLIME/colx585_group_project_data` dataset is designed and a continuation of our Membership Inference Attacks (MIA), but with the addition of Vision-Language Models. Each of the samples we have viewed on `milestone4/document/data_inspection.md` consists of five features: a unique `id`, an `image` (stored as raw bytes), the target `is_member` binary label (0 or 1), a `text` string containing a conversational prompt and response, and a `type` category (e.g., `seen_img_unseen_txt`). The `type` feature is particularly interesting and useful as it seems to categorize if the image, text, or both has been seen during training which may be very useful.

### Descriptive Statistics

*Note: The text length calculations represent the character count of the combined prompt and response string.*

| Split | Number of samples | Mean Text Length | Median Text Length | Max length | Min length | Length Std Dev |
|:----------|:----------|:----------|:----------|:----------|:----------|:----------|
| train | 6,000 | 261.93 | 246 | 813 | 98 | 89.24 |
| validation | 1,200 | 258.05 | 241 | 737 | 104 | 86.61 |
| test | 6,000 | 247.19 | 232 | 1,079 | 82 | 91.35 |

### Literature Review

**Paper 1: Membership Inference Attacks against Large Vision-Language Models**

This paper addresses the important data security concerns regarding VLLMs' training process. The inclusion of sensitive private information such as private photos or medical records can be a very worrying issue. And how to detect such data is still a unresolved issue because of lack of datasets and right methodologies. In the paper, they introduced benchmark to facilitate training data detection. They also included a novel MIA pipeline designed for token-level image detection and a whole new metric.

**Short summary:**

**Main Task**: Detecting whether a specific image or text was used to train a vision-language model (i.e., membership inference attack on VLLMs). This is framed as a binary classification problem — given a data point, determine if it's a member or non-member of the training set.

**Dataset**: The authors constructed **VL-MIA**, which includes three sub-datasets: VL-MIA/DALL-E (592 member images from LAION-CCS vs. DALL-E-generated non-member images), VL-MIA/Flickr (600 MS COCO member images vs. recent Flickr photos as non-members), and VL-MIA/Text (600 samples of instruction-tuning text vs. GPT-4-generated answers as non-members). They also used the existing **WikiMIA** benchmark for LLM pre-training text detection.Model: The study evaluated the Pythia model series (160m to 12b parameters) and the OLMo series (1b to 13b parameters).

**Models:** Three open-source VLLMs — **LLaVA 1.5**, **MiniGPT-4**, and **LLaMA-Adapter V2**. They also tested on closed-source **GPT-4** (vision-preview API).

**Main Contribution:** They released the first MIA benchmark (VL-MIA) specifically designed for VLLMs and they also proposed a **cross-modal pipeline** that enables image MIA by computing metrics from text logit slices, solving the problem that image tokens aren't directly accessible in VLLMs. They finally introduced **MaxRényi-K%**, a target-free metric based on Rényi entropy that generalizes existing methods like Min-K% and works on both image and text modalities.

**Paper 2: The Sample Complexity of Membership Inference and Privacy Auditing**

This paper studies membership inference attacks (MIAs) from a theoretical angle by asking how much extra information an attacker needs to succeed. Instead of testing LLMs or VLMs directly, it looks at a simpler setting called Gaussian mean estimation, where a model learns from $n$ samples and releases a noisy estimate. The main finding is that an attacker may need far more reference samples than the model used for training, unlike many existing MIA methods that only use $O(n)$ samples. The paper also points out that current privacy audits may underestimate risk if they do not fully use information about the data distribution.
**Short summary:**

**Main Task**: The paper tries to solve the problem of sample complexity in membership inference attacks. More specifically, it asks: given access to a model output and a target example, how many auxiliary samples from the same population are needed for an attacker to distinguish whether the target was in the training set or not? The paper focuses on sample-based MIAs in Gaussian mean estimation and compares the unknown-covariance and known-covariance settings.

**Dataset**: This paper does not use a real benchmark dataset like Flickr, LAION, or WikiMIA. Instead, it studies a theoretical data setting where the population distribution is a $d$-dimensional Gaussian distribution $N(\mu, \Sigma)$, and the training data consists of $n$ i.i.d. samples drawn from that distribution.

**Models:** Model: The model studied in the paper is not a VLM or LLM, but a Gaussian mean estimator. The main estimator considered is the noisy empirical mean, which outputs the sample mean plus Gaussian noise: $$\\hat{\\mu} = \\frac{1}{n} \\sum_{i=1}^{n} X_i + \\rho Z, Z \\sim N(0, \\Sigma)$$
The attacker then tries to infer whether a target sample was used in training based on this released estimate and a set of auxiliary samples.

**Main Contribution:** The paper shows that when the covariance is unknown, a successful sample-based MIA may need much more auxiliary data than expected, up to $\\Omega(n + n^2\\rho^2)$ samples. This means the attacker can sometimes need far more data than the model used for training. It also shows that the problem becomes much easier when the covariance is known, so estimating covariance is the main challenge. Overall, the paper suggests that many current MIA methods, which usually use only $O(n)$ reference samples, may underestimate real privacy risk.

***Citation:** Li, Z., Wu, Y., Chen, Y., Tonin, F., Abad Rocamora, E., & Cevher, V. (2024). Membership Inference Attacks against Large Vision-Language Models. Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024).*

***Citation:** Haghifam, M., Smith, A., & Ullman, J. (2025). The Sample Complexity of Membership Inference and Privacy Auditing. arXiv preprint arXiv:2508.19458.*

---

## Contributions

### Milestone 4

| Member       | Percentage |
|--------------|------------|
| Tianhao Cao  | 25%        |
| Yusen Huang  | 25%        |
| Marco Wang   | 25%        |
| Darwin Zhang | 25%        |

**Total:** 100%

