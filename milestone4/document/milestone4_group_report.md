## Data

### High-Level Description
The `UBC-SLIME/colx585_group_project_data` dataset is designed and a continuation of our Membership Inference Attacks (MIA), but with the addition of Vision-Language Models. Each of the samples we have viewed on `milestone4/document/data_inspection.md` consists of five features: a unique `id`, an `image` (stored as raw bytes), the target `is_member` binary label (0 or 1), a `text` string containing a conversational prompt and response, and a `type` category (e.g., `seen_img_unseen_txt`). The `type` feature is particularly interesting and useful as it seems to categorize if the image, text, or both has been seen during training which may be very useful.

### Descriptive Statistics
*Note: The text length calculations represent the character count of the combined prompt and response string.*

| Split | Number of samples | Mean Text Length | Median Text Length | Max length | Min length | Length Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| train | 6,000 | 261.93 | 246 | 813 | 98 | 89.24 |
| validation | 1,200 | 258.05 | 241 | 737 | 104 | 86.61 |
| test | 6,000 | 247.19 | 232 | 1,079 | 82 | 91.35 |