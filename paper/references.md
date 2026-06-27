# References

All papers cited or considered for the paper.

---

## Humor Datasets / Shared Tasks

| Paper | Link | Notes |
|-------|------|-------|
| Hessel et al. (2023), *Do Androids Laugh at Electric Sheep? Humor "Understanding" Benchmarks from The New Yorker Caption Contest* | https://arxiv.org/abs/2209.06293 | NYCC dataset. CrowdAcc=83.7%, NYAcc=64.6% (human ceiling we use). Native pairwise format. |
| Meaney et al. (2021), *SemEval-2021 Task 7: HaHackathon, Detecting and Rating Humor and Offense* | https://aclanthology.org/2021.semeval-1.9 | HaHackathon dataset. 10,000 texts (Twitter + Kaggle jokes), 20 annotators per text, rated 0–5. Task 1a winner F1=0.9854, Task 1b winner RMSE=0.4959. No pairwise subtask — our pairwise eval is synthetic. |
| Hossain et al. (2020), *SemEval-2020 Task 7: Assessing Humor in Edited News Headlines* | https://aclanthology.org/2020.semeval-1.98/ | Humicroedit dataset. Subtask 1: regression (0–3). Subtask 2: pairwise. 48/31 teams. Best pairwise acc = 67.43% (Hitachi). Baseline 49.0%. |
| Potash et al. (2017), *SemEval-2017 Task 6: #HashtagWars: Learning a Sense of Humor* | https://aclanthology.org/S17-2004/ | Pairwise humor ranking on @midnight tweets (crowd votes = ground truth). 7 teams; best pairwise acc 67.5% (HumorHawk). Structurally identical to our task. |
| Castro et al. (2018), *A Crowd-Annotated Spanish Corpus for Humor Analysis* | https://aclanthology.org/W18-3502/ | HAHA 2018. Inter-annotator agreement: alpha=0.5710 for binary humor detection, alpha=0.1625 for funniness rating (authors describe as "closer to random annotation"). Predecessor to HAHA 2019. |
| Chiruzzo et al. (2019), *HAHA 2019: Humor Analysis Based on Human Annotation* | https://www.fing.edu.uy/inco/grupos/pln/haha/2019/ | HAHA Spanish dataset. Spanish tweets rated 1–5 by screened crowd annotators. 24k train / 6k test. Inter-annotator agreement alpha=0.224 for funniness rating. |
| Weller & Seppi (2019), *Humor Detection: A Transformer Gets the Last Laugh* | https://github.com/orionw/Reddit-Humor | Reddit jokes dataset. Original use: binary humor detection. We repurpose upvote ratio as continuous funniness signal for pairwise ranking. |
| Ethayarajh et al. (2022), *Understanding Dataset Difficulty with V-Usable Information* | https://arxiv.org/abs/2110.08420 | SHP dataset (Stanford Human Preferences). Better Reddit comment A or B? Upvote-based preference. ICML 2022 Outstanding Paper. |

---

## SemEval Systems / Winners

| Paper | Link | Notes |
|-------|------|-------|
| Cano Berlanga et al. (2017), *HumorHawk at SemEval-2017 Task 6* | https://aclanthology.org/S17-2010/ | 2017 Task 6 winner (67.5%). GloVe + phonetic embeddings → LSTM + char-CNN + XGBoost ensemble. Used sound/phonetics as humor signal. |
| Hitachi (2020), *Stacking at Scale with Heterogeneous Language Models for Humor Recognition* | https://aclanthology.org/2020.semeval-1.101/ | 2020 Task 7 winner. RMSE 0.497 (subtask 1), pairwise acc 67.43% (subtask 2). BERT+GPT-2+RoBERTa+XLNet stacking ensemble. |
| SemEval-2026 Task 1, *MWAHAHA: Multimodal Wit and Humor for Automated Humor Assessment* | https://arxiv.org/abs/2606.00022 | Uses Bradley-Terry preference model. "lmfaoooo" system: 17-dim humor basis scored by LLM → logistic regression. Pointwise variant 45–63%. Dimensions: Clear Punchline, Wordplay, Universality, Subtlety, Avoid Cliché, Fresh Perspective, Exaggeration, Subverting Expectations, Character-Driven, Economy of Words, Self-Deprecation, Satirical Edge, Anthropomorphism, Clever Analogies, Memorable Imagery, Dark Humor, Natural Dialogue. |

---

## LLM-as-Judge / Evaluation

| Paper | Link | Notes |
|-------|------|-------|
| Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | https://arxiv.org/abs/2306.05685 | Foundational LLM-as-judge paper. GPT-4 agrees with humans ~80% on open-ended tasks. Key failure modes: position bias, verbosity bias, self-enhancement bias. Subjective/cultural tasks (humor) are known failure cases. |
| Verga et al. (2024), *A Survey on LLM-as-a-Judge* | https://arxiv.org/abs/2411.15594 | Known biases: self-enhancement, verbosity, position. GPT-4 agrees with humans ~80%. Alignment degrades on subjective/cultural tasks like humor. |
| Wataoka et al. (2024), *Self-Preference Bias in LLM-as-a-Judge* | https://arxiv.org/abs/2410.21819 | Self-preference bias from perplexity familiarity — models prefer outputs stylistically similar to themselves. |
| Wang et al. (2025), *Improving LLM-as-a-Judge Inference with the Judgment Distribution* | https://arxiv.org/abs/2503.03064 | EMNLP 2025. CoT collapses judgment distribution, hurts LLM-as-a-judge in 30/40 scoring cases. Corroborates our SHP CoT finding (-3.4pp). |

---

## Crowd Preferences / Persona Simulation

| Paper | Link | Notes |
|-------|------|-------|
| Goes et al. (2022), *Crowd Score: A Method for the Evaluation of Jokes using LLM AI Voters* | https://arxiv.org/abs/2212.11214 | 4 HSQ humor personality types (affiliative/self-enhancing/aggressive/self-defeating) vote Funny/Boring. Validated on only 52 jokes. Our empirical result: ~53% on HaHa pairwise (n=1400) — essentially random. |
| Santurkar et al. (2023), *Whose Opinions Do Language Models Reflect?* | https://arxiv.org/abs/2303.17548 | OpinionsQA. LLM opinion distributions misalign with actual subgroup views. Persona prompting has low steerability. Models default to culturally dominant viewpoints. Directly undermines Goes et al. |
| Kirk et al. (2024), *The PRISM Alignment Dataset* | https://arxiv.org/abs/2404.16019 | 1,500 participants, 75 countries. Human preferences are diverse and cross-culturally inconsistent. Supports why crowd aggregation is needed. |
| Zheng et al. (2023), *Chatbot Arena* | https://arxiv.org/abs/2403.04132 | Crowdsourced human preference benchmark — gold standard for real human preference data. |

---

## Fine-tuning / Representation

| Paper | Link | Notes |
|-------|------|-------|
| Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* | https://arxiv.org/abs/2106.09685 | Parameter-efficient fine-tuning. We use r=16, alpha=32. |
| Ouyang et al. (2022), *Training language models to follow instructions with human feedback (InstructGPT)* | https://arxiv.org/abs/2203.02155 | Pairwise preference training paradigm (RLHF). We use a simpler in-batch BCE pairwise loss without PPO. |
| Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency* | https://arxiv.org/abs/2310.01405 | RepE / LAT. Concepts like emotions are linear directions in LLM activation space. Prompt template "Consider the amount of [concept] in the following: [text]" elicits the concept reliably. Directly inspires our funniness direction probe. |
| Anthropic Interpretability Team (2026), *Emotion Concepts in Claude* | https://transformer-circuits.pub/2026/emotions/index.html | Emotion vectors in Claude Sonnet 4.5. Same spirit as RepE — confirms emotions are linearly represented in frontier models. |

---

## Cross-dataset Humor Transfer

| Paper | Link | Notes |
|-------|------|-------|
| Baranov et al. (2023), *hri_tools humor dataset bundle* | TBD — find arxiv link | Trained on multiple humor datasets, evaluated cross-dataset; diversity aids generalization. All 11 datasets are binary-labeled (no continuous crowd preference signal). We differ: pairwise preference objective, continuous crowd labels. |
