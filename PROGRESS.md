# Progress Log
**Goal**: Paper by ~2026-06-24 (flexible)

This file tracks experiments, results, and key references needed for writing the final paper.
Update the log section after each experiment.

---

## Paper Direction

We are building a collection of empirical findings about fine-tuning LLMs for humor preference alignment. Basically sort of what we discover along the way of trying to make LLMs humorous. A key question is what can be done to give LLM's sort of a taste of humor or even a good sense of humor.

The original mechanism hypothesis (self-preference bias → style shift → preference shift via De-GPT DPO) was explored but found weak: DPO produced no meaningful improvement on NYCC or SHP, and Discord SFT hurt performance across the board. The paper is now a findings paper — reporting what works, what doesn't, and why, across humor and general preference datasets.

**Current direction**: All zero-shot and prompt-engineering approaches plateau at ~54–57% on humor pairwise tasks. SemEval winners (2017, 2020, 2021) all achieve 67%+ through supervised training on crowd-labeled humor data — that gap is the key finding. Next step: build a supervised humor preference model trained on combined crowd-labeled data (NYCC + HaHa + Humicroedit), then use it as a reward signal in a demo pipeline (funny LLM in group chat / conversational agent).

**Original hypothesis (abandoned)**:
Fine-tune on human-style text (Discord SFT / De-GPT DPO) → model generation style shifts → self-preference bias causes preference judgments to shift toward human-majority taste. Abandoned because: (1) DPO showed no effect, (2) Discord SFT degradation may be general reasoning degradation rather than preference-specific, (3) NYCC captions are both human-written so the self-preference mechanism doesn't cleanly apply.
The reason i think is because overall understanding decreased after finetuning and i dont have the skills to do advanced stuffs? but cant verify it anyways.
---

## Key Findings

1. **Discord SFT causes general capability degradation**, not humor-specific degradation. Confirmed across all three datasets: NYCC (-5pp Hermes), HaHa (r=0.228→0.015), and SHP (-3.7pp: hermes 59.5% → discord-hermes 55.8%). Naively fine-tuning on casual human conversation (Discord) without a targeted method hurts general reasoning/understanding, which then hurts everything downstream. This rules out the "preference shift" mechanism — the model isn't learning human taste, it's just getting dumber.

2. **De-GPT DPO has no effect on any dataset** — NYCC -1.0pp, SHP +0.6pp, HaHa pairwise -0.6pp, Jester pairwise +0.2pp. All within ±2.2pp noise. DPO on human-vs-AI style data does not shift humor preference.

3. **All base models plateau at ~53–56% on NYCC** regardless of architecture (Gemma4, Qwen3.5, Hermes-3). Human ceiling is ~64.6% (NYAcc). The gap is large and no fine-tuning approach tested has closed it.

4. **CoT does not improve — and may hurt — LLM judgment on subjective preference tasks.** SHP: -3.5pp (significant, n=2000). NYCC: -1.3pp (within noise, n=2000). Neither dataset benefits from reasoning. Note: "thinking" here is prompt-level CoT, not built-in reasoning tokens. Consistent with Wang et al. (2025) who found CoT collapses judgment distribution in 30/40 scoring cases.
   - **Interpretation**: Subjective preference tasks (humor, Reddit upvotes) have no ground-truth reasoning path — there is no correct logic for why something is funnier or more upvoted. CoT either hurts by over-rationalising or provides no gain. This is a broader claim than humor-specific and holds across both datasets tested.

5. **Prompt framing makes no difference** across all models and datasets — gut vs no-gut, and personal taste vs crowd prediction ("which would a crowd find funnier?") all produce identical results within noise (±2.2pp). The model's humor judgment is stable regardless of how the question is framed.

---

## Datasets
| Name | Description | Split used | Size | Paper | Access |
|------|-------------|-----------|------|-------|--------|
| NYCC (New Yorker Caption Contest) | Funnier caption A or B? Crowd-aggregated ground truth | 5-fold validation | up to 2616 | Hessel et al. (2022) | https://arxiv.org/abs/2209.06293 |
| SHP (stanfordnlp/SHP) | Better Reddit comment A or B? Upvote-based preference, score_ratio ≥ 2 | validation | 2000 | Ethayarajh et al. (2022) | https://arxiv.org/abs/2110.08420 |
| HaHackathon (SemEval-2021 Task 7) | 1000 English jokes each rated 0–5 by multiple AMT crowd annotators | test set | 1000 jokes | Meaney et al. (2021) | https://aclanthology.org/2021.semeval-1.9 |
| Jester (dataset 3) | 150 English jokes rated -10 to +10 by 54,905 users; 140 usable; avg ratings -2.74 to +3.66 | full dataset | 140 usable jokes | Goldberg et al. (2001) | https://eigentaste.berkeley.edu/dataset/ |
| HAHA Spanish Twitter (2019) | Spanish tweets rated 0–4 for humor by screened crowd annotators; individual vote counts (votes_1–5) + funniness_average available | train+test | 24k train (9,253 humorous + 14,747 non-humorous) / 9k test | Chiruzzo et al. (2019) | https://www.fing.edu.uy/inco/grupos/pln/haha/2019/ |
| Humicroedit / FunLines | ~15,000 English news headlines with word edits rated 0–3 for funniness by 5 AMT judges each; SemEval-2020 Task 7. FunLines (8,248 rows) already on disk at datasets/humicroedit/semeval-2020-task-7-dataset/subtask-1/train_funlines.csv | train+dev+test | 9652 train / 1757 probe subset (957 funny ≥2.0, 799 unfunny =0.0) | Hossain et al. (2020) | https://arxiv.org/abs/2008.00304 |
| Oogiri-GO / Oogiri-Corpus | Japanese humor task; ~100 candidate responses per prompt each rated by ~100 independent blind judges — eliminates popularity bias | TBD | TBD — not yet downloaded | Murakami et al. / Zhong et al. | https://arxiv.org/abs/2512.21494 |
| Yelp Funny Reviews | Yelp reviews with crowd "funny" votes; general preference signal similar to SHP — useful if we expand beyond humor | TBD | Large — not yet sourced | de Oliveira & Rodrigo (Stanford CS224d) | No public repo |
| Cards Against Humanity (CAH Lab) | Real player gameplay choices — naturalistic behavioral ground truth, not paid annotation. **Not currently obtainable** — requires emailing lab | N/A | N/A | Cards Against LLMs | N/A — email lab |
| Open Mic Dataset | Stand-up comedy transcripts (~40 hrs); humor quotient derived from real audience laughter detection — behavioral ground truth, not annotation; validated vs 3 annotators (kappa=0.6) | N/A | N/A — not accessible | Mittal et al. | https://arxiv.org/abs/2110.12765 |
| Discord-Dialogues | SFT training — casual human multi-turn chat | first 50,000 | 50,000 | — | mookiezi/Discord-Dialogues (HF) |
| De-GPT-DPO | DPO training — human (chosen) vs AI (rejected) text pairs | first 5,000 | 5,000 | — | qingy2024/De-GPT-DPO (HF) |

Human baselines:
- NYCC CrowdAcc: 83.7% (humans predicting crowd preference)
- NYCC NYAcc: 64.6% (humans predicting editor choice)
- Our eval mix: ~71% official_winner / ~29% crowd_winner → relevant ceiling ≈ 64.6%
- NYCC unique crowd_winner examples: 757 total across all folds

SHP: preferences inferred from Reddit upvotes (no IAA reported). score_ratio ≥ 2 filter for cleaner signal.

---

## Eval Setup

- All evals: `do_sample=False`, greedy decoding — fully deterministic
- 95% CI at n=2000: ±2.2pp | at n=1000: ±3.1pp | at n=757: ±3.6pp
- Going forward: **plain (no-gut)** prompt only

**Prompt modes tested** (gut vs no-gut confirmed identical — never run both again):
- **gut**: `"Use your gut feeling and return <answer>A</answer> or <answer>B</answer>."`
- **plain (no-gut)**: `"Return <answer>A</answer> or <answer>B</answer>."`
- **thinking (CoT)**: `"Briefly explain why each is good or bad, then return your final choice as <answer>A</answer> or <answer>B</answer>."` — prompt-level CoT only, no built-in reasoning tokens (`enable_thinking=False`)

**CoT thinking results**:
| Dataset | Model | plain | CoT | delta |
|---------|-------|-------|-----|-------|
| NYCC | qwen3.5:4b | 54.5% | 53.2% (n=2000) | -1.3pp (within noise) |
| NYCC | qwen4b-degpt-dpo | 53.5% | 52.2% (n=1000) | -1.3pp (underpowered) |
| SHP | qwen3.5:4b | 63.2% | 59.8% (n=2000) | **-3.4pp** |
| SHP | qwen4b-degpt-dpo | 63.8% | 59.7% (n=2000) | **-3.7pp** |

NYCC CoT result at n=2000: -1.3pp (within ±2.2pp CI) — no significant effect. Confirms humor judgment is not hurt by CoT, unlike SHP (-3.5pp). Theory: crowd humor signal has no reasoning path to anchor to, so CoT neither helps nor hurts.

---

## Results
| Model | Training | NYCC (plain) | SHP (plain) | HaHa (Spearman r) | HaHa pairwise | Jester pairwise |
|-------|----------|--------------|-------------|-------------------|---------------|-----------------|
| gemma4:e4b | base | 53.7% | — | — | — | — |
| gemma4-e4b-discord | SFT (discord) | 50.6% | — | — | — | — |
| qwen3.5:4b | base | 54.5% (n=2616) | 63.2% (n=2000) | — | 55.4% (n=2000) | 55.2% (n=2000) |
| qwen4b-degpt-dpo | DPO (De-GPT) | 53.5% (n=1000) | 63.8% (n=2000) | — | 54.8% (n=2000) | 55.4% (n=2000) |
| hermes-3-8b | base (SFT+DPO synthetic) | 55.6% | 59.5% (n=2000) | 0.228 (n=671) | — | — |
| discord-hermes-3-8b | + Discord SFT | 50.6% | 55.8% (n=2000) | 0.015 (n=1000) | — | — |
| llama-3.1-8b-instruct | RLHF-aligned | ~50% | — | 0.277 (n=971) | — | — |

---

## TODO
1. ~~**Run qwen3.5:4b CoT on NYCC n=2000**~~ ✓ Done — 53.2% CoT vs 54.5% plain, -1.3pp within noise. CoT has no significant effect on humor judgment (unlike SHP -3.5pp). Humor-is-intuitive theory supported.
2. ~~**Run discord-hermes-3-8b on SHP**~~ ✓ Done — hermes 59.5% vs discord-hermes 55.8%, degradation is general
3. ~~**Collect humor datasets**~~ ✓ Decided: eval trio = NYCC + HaHackathon + Jester. All already on disk.
4. ~~**Build Jester preprocessing script**~~ ✓ Done — 2000 pairs, avg gap 3.13, 50/50 A/B balance
5. ~~**Eval qwen3.5:4b vs qwen4b-degpt-dpo on HaHackathon + Jester**~~ ✓ Done — DPO flat on all datasets. Conclusion: De-GPT DPO does not shift humor preference.
6. ~~**Decide next step**~~ ✓ DPO conclusively flat across all 4 datasets. Pivoting to reward model.
7. **Reward model** — train a discriminative head on humor preference data (NYCC + HaHa + Jester). No next-token generation, just score output. Connects to CoT finding (verbalization doesn't help for subjective tasks).

---

## References
| Paper | Link | Key takeaway |
|-------|------|--------------|
| Wataoka et al. (2024), *Self-Preference Bias in LLM-as-a-Judge* | https://arxiv.org/abs/2410.21819 | Self-preference bias from perplexity familiarity — models prefer outputs stylistically similar to themselves |
| Verga et al. (2024), *A Survey on LLM-as-a-Judge* | https://arxiv.org/abs/2411.15594 | Known biases: self-enhancement, verbosity, position; GPT-4 agrees with humans ~80%; alignment degrades on subjective/cultural tasks like humor |
| Zheng et al. (2023), *Chatbot Arena* | https://arxiv.org/abs/2403.04132 | Crowdsourced human preference benchmark — gold standard for real human preference data |
| Hessel et al. (2022), *Do Androids Laugh at Electric Sheep?* | https://arxiv.org/abs/2209.06293 | NYCC dataset; CrowdAcc=83.7%, NYAcc=64.6% |
| Ethayarajh et al. (2022), *Understanding Dataset Difficulty with V-Usable Information* | https://arxiv.org/abs/2110.08420 | SHP dataset paper (ICML 2022 Outstanding Paper) |
| Kirk et al. (2024), *The PRISM Alignment Dataset* | https://arxiv.org/abs/2404.16019 | 1,500 participants, 75 countries — human preferences are diverse and cross-culturally inconsistent; supports why crowd aggregation is needed |
| Wang et al. (2025), *Improving LLM-as-a-Judge Inference with the Judgment Distribution* | https://arxiv.org/abs/2503.03064 | EMNLP 2025; CoT collapses judgment distribution, hurts LLM-as-a-judge in 30/40 scoring cases — corroborates our SHP CoT finding |
| Potash et al. (2017), *SemEval-2017 Task 6: #HashtagWars: Learning a Sense of Humor* | https://aclanthology.org/S17-2004/ | Pairwise crowd humor ranking on @midnight tweets (crowd votes = ground truth). 7 teams; best pairwise accuracy 67.5% (HumorHawk). Structurally identical to our task. |
| Cano Berlanga et al. (2017), *HumorHawk at SemEval-2017 Task 6* | https://aclanthology.org/S17-2010/ | 2017 Task 6 winner (67.5%): GloVe + phonetic embeddings → LSTM + char-CNN + XGBoost ensemble. Notably used sound/phonetics as humor signal. |
| Hossain et al. (2020), *SemEval-2020 Task 7: Assessing Humor in Edited News Headlines* | https://aclanthology.org/2020.semeval-1.98/ | Humicroedit task paper. Subtask 1: regression (RMSE, 0–3 scale). Subtask 2: pairwise — which of two edits of the same headline is funnier (accuracy). 48/31 teams. Verified results: baseline 49.0%, best benchmark RoBERTa+FT 65.0%, winner Hitachi 67.43% (+17.9pp over baseline). Hitachi used BERT+GPT-2+RoBERTa+XLNet stacking ensemble; subtask 2 = pick whichever edit scores higher on subtask 1 regression model. |
| Hitachi (2020), *Stacking at Scale with Heterogeneous Language Models for Humor Recognition* | https://aclanthology.org/2020.semeval-1.101/ | 2020 Task 7 winner (all subtasks). RMSE 0.497 (subtask 1), pairwise acc 67.43% (subtask 2). |
| Meaney et al. (2021), *SemEval-2021 Task 7: HaHackathon, Detecting and Rating Humor and Offense* | https://aclanthology.org/2021.semeval-1.9 | HaHackathon dataset + task paper. 10,000 texts (Twitter + Kaggle jokes), 20 annotators per text, rated 0–5 (asked only if judged humorous). No pairwise subtask — our pairwise eval is synthetic. **Task 1a (binary humor detection)**: winner PALI F1=0.9854, BERT baseline F1=0.9283. **Task 1b (humor rating RMSE, lower=better)**: winner abcbpc RMSE=0.4959 (Kaggle 0.4544 / Twitter 0.5141), BERT baseline RMSE=0.8000 overall. **Task 1c (controversy)**: winner PALI F1=0.6302, BERT baseline F1=0.6232. Top systems: pre-trained LM ensembles (BERT/RoBERTa/DeBERTa/ALBERT) with multi-task learning + adversarial training. |
| Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency* | https://arxiv.org/abs/2310.01405 | RepE / LAT: concepts like emotions are linear directions in LLM activation space; can both READ (probe) and WRITE (steer) them. Prompt template `"Consider the amount of [concept] in the following: [text]"` elicits the concept reliably. Directly inspires our funniness direction probe. |
| Anthropic Interpretability Team (2026), *Emotion Concepts in Claude* | https://transformer-circuits.pub/2026/emotions/index.html | Finds emotion vectors in Claude Sonnet 4.5 by generating stories for 171 emotion words and recording activations — same spirit as RepE. Confirms emotions are linearly represented in frontier models. |
| SemEval-2026 Task 1, *MWAHAHA: Multimodal Wit and Humor for Automated Humor Assessment* | https://arxiv.org/abs/2606.00022 | Uses Bradley-Terry preference model (same as our approach); "interpretable pipeline" = 17-dim humor basis scored by LLM → logistic regression. Pure prompting, not representation-level. 17 dimensions: Clear Punchline, Wordplay, Universality, Subtlety, Avoid Cliché, Fresh Perspective, Exaggeration, Subverting Expectations, Character-Driven, Economy of Words, Self-Deprecation, Satirical Edge, Anthropomorphism, Clever Analogies, Memorable Imagery, Dark Humor, Natural Dialogue. |
| Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | https://arxiv.org/abs/2306.05685 | Foundational LLM-as-judge paper. GPT-4 agrees with humans ~80% on open-ended tasks. Named key failure modes: position bias, verbosity bias, self-enhancement bias. Subjective/cultural tasks (humor) are known failure cases. |
| Goes et al. (2022), *Crowd Score: A Method for the Evaluation of Jokes using LLM AI Voters* | https://arxiv.org/abs/2212.11214 | Proposes LLM personas (4 humor styles: affiliative/self-enhancing/aggressive/self-defeating) as humor judges. Validated on 52 jokes — too small to generalize. Aggregate score correlates with human funniness directionally. Our counter: 54–57% zero-shot already shows LLM humor judgment is weak; persona simulation doesn't fix it (Santurkar et al.). |
| Santurkar et al. (2023), *Whose Opinions Do Language Models Reflect?* | https://arxiv.org/abs/2303.17548 | OpinionsQA benchmark, 60 US demographic groups. LLM opinion distributions substantially misalign with actual subgroup views. Persona prompting has low steerability. Models default to culturally dominant viewpoints. Directly undermines Goes et al. persona simulation claim. |

---

## Models
| Name | Checkpoint / HF ID | Chat template | Type |
|------|--------------------|---------------|------|
| qwen3.5:4b | unsloth/Qwen3.5-4B | qwen-3 | Base (HF) |
| qwen4b-degpt-dpo | outputs/qwen4b-degpt-dpo/checkpoint-625 | qwen-3 | DPO fine-tune |
| hermes-3-8b | NousResearch/Hermes-3-Llama-3.1-8B | chatml | Base (HF) |
| discord-hermes-3-8b | mookiezii/Discord-Hermes-3-8B | chatml | SFT fine-tune |
| llama-3.1-8b-instruct | meta-llama/Llama-3.1-8B-Instruct | llama-3.1 | Base (HF) |
| gemma4-e4b-discord | outputs/gemma4-e4b-discord/checkpoint-4011 | gemma-4 | SFT fine-tune |

---

## Training
- **DPO**: `qingy2024/De-GPT-DPO` — 5000 examples, 1 epoch
  - Checkpoint: `outputs/qwen4b-degpt-dpo/checkpoint-625`
  - Token lengths verified: prompt avg 18 tokens, chosen avg 104 — no truncation issues

---

## Log
| Date | Action |
|------|--------|
| 2026-06-09 | Ran base qwen3.5:4b on SHP (all 3 prompt variants). Fixed summary.csv path in run_eval_shp.py. |
| 2026-06-12 | Confirmed negative result. Verified DPO training bug was not an issue. Identified perplexity gap as missing experiment. |
| 2026-06-13 | HaHackathon rating eval: Hermes r=0.228, Discord-Hermes r=0.015, Llama r=0.277. Discord SFT destroys humor correlation — confirmed across NYCC + HaHa. |
| 2026-06-17 | Confirmed gut vs no-gut makes zero difference. CoT prompting hurts SHP ~3.5pp (n=2000) but not NYCC (n=1000, underpowered). Abandoned original hypothesis. Restructured PROGRESS.md. |
| 2026-06-22 | Pivoted to activation probing. Trained logistic regression probe on Humicroedit (957 funny ≥2.0, 799 unfunny =0.0). Best layer 16: 70.9% CV. Cross-domain on HaHa pairwise (synthetic): 55.5%. Funniness direction exists linearly but is domain-specific. |
| 2026-06-22 | Changed eval prompt to crowd-framing ("which would a crowd find funnier if shared online?"). Running on NYCC + HaHa pairwise for qwen3.5:4b. Renamed run_eval.py → run_eval_nycc.py. |
| 2026-06-23 | Crowd framing results in: NYCC 54.2% (vs plain 54.5%, -0.3pp), HaHa 54.4% (vs 55.4%, -1.0pp). Both within noise. Null result — extends finding 5: prompt framing (gut/plain/crowd) has no effect on model humor judgment. |
| 2026-06-23 | Humicroedit pairwise (subtask-2 test, ties excluded): qwen3.5:4b 56.3% (n=2628). Baseline 49.0%, 2020 winner (Hitachi) 67.43%. Zero-shot LLM sits 7pp above baseline but 11pp below fine-tuned winner. |
| 2026-06-23 | Humicroedit few-shot rating (5 anchors from train, 0–3 scale): qwen3.5:4b RMSE=0.6726 (n=1701) vs Hitachi 0.497. Derived pairwise: 57.1% (n=1500) vs direct 56.3% — negligible gain. Model collapses toward 0.5 on uncertain items. Rating-then-ranking gives no benefit over direct pairwise prompting. |
| 2026-06-23 | HaHa few-shot rating (5 anchors from pairwise.csv, 0–5 scale, n=1479 jokes): qwen3.5:4b RMSE=1.3055, Spearman r=0.191, MAE=1.104. Derived pairwise 56.3% (n=985) vs direct 54.4% — within noise. RMSE worse than 2021 BERT baseline (0.8000). Model collapses to anchor values (mostly 1.2). Conclusion: few-shot rating fails; LLM cannot spread predictions across humor scale. Note: rating.csv and pairwise.csv IDs only 487/2376 overlap — must load jokes from pairwise.csv directly. |
| 2026-06-23 | Reward model (qwen4b-humor-reward): 52.4% on NYCC (n=2616), 49.9% on HaHa reward test (n=2831) — both near random. Cause unclear: possibly insufficient training signal, wrong training data framing, or humor too abstract for discriminative head trained this way. |
| 2026-06-23 | Research direction clarified: all zero-shot/prompt approaches plateau at 54–57%. SemEval winners achieve 67%+ via supervised training on crowd labels. Next: supervised humor preference model on combined crowd data → demo pipeline. |
| 2026-06-24 | Humor basis pointwise (lmfaoooo 17-feature approach, pointwise variant): Humicroedit subtask-2, L1=56.1%, L2=56.3% (n=1000 test pairs). No improvement over zero-shot 56.3%. Matches lmfaoooo paper prediction (their pointwise was 45–63%). Full feature weights saved to results/humicroedit/basis_weights_20260623_214650.csv. Pairwise feature scoring (their winning variant) not yet implemented. |
| 2026-06-24 | Pairwise crowd probe (h_A - h_B, 3000 train / 1000 test): best 56.7% (layer 20) — essentially zero-shot level. Direct crowd pairwise supervision fails to find a useful direction; noisy pairwise signal is the issue. Binary funny/unfunny probe (61.9%) is clearly better. |
| 2026-06-24 | Margin-filtered pairwise probe (MIN_MARGIN=1.0, only pairs where |meanGrade_A - meanGrade_B| >= 1.0): **67.0%** at layer 16 (n=612 train / 204 test pairs after filter). Beats frozen binary probe 61.9% by +5.1pp, LoRA regression 64.8% by +2.2pp, and matches Hitachi 2020 winner 67.43% within noise. Cleaner pairwise signal (large margin) dramatically improves probe. |
| 2026-06-24 | LoRA regression (fine-tune Qwen3.5-4B with LoRA r=16 + regression head, MSE on subtask-1 meanGrade, 9652 examples, 3 epochs): **64.8%** on subtask-2 pairwise (n=2628). Beats frozen binary probe 61.9% by +2.9pp. Training loss: 0.341 → 0.246 → 0.184. Saved to results/probe/cache/lora_regression/. |
| 2026-06-24 | Activation probe (Humicroedit subtask-2 pairwise): layer 16 = **61.9%** (n=2628 pairs, ties excluded). Beats zero-shot 56.3% by +5.6pp and humor basis 56.3% by +5.6pp. Probe was trained only on binary funny/unfunny labels (not pairwise) — the funniness direction generalises to pairwise ranking. Per-layer: L8=59.4%, L12=60.0%, L16=61.9%, L20=61.7%, L24=61.6%. Results saved to results/probe/probe_humicro_20260624_111557.csv. |
| 2026-06-22 | Surveyed SemEval humor tasks for baselines. Key verified number: SemEval 2020 Task 7 (Humicroedit) best pairwise acc = 67.43% (Hitachi, fine-tuned ensemble). Baseline = 49.0%. 2017 Task 6 results TBD (check PDF). 2021 Task 7 verified from PDF: 1a winner PALI F1=0.9854, 1b winner abcbpc RMSE=0.4959, no pairwise subtask confirmed. Added SemEval papers to References. |
| 2026-06-24 | HaHa LoRA regression (same method as Humicroedit: LoRA r=16 + regression head, MSE on humor_rating 0–5, 3945 train jokes, 3 epochs): **68.3%** on HaHa pairwise held-out test (n=2000, proper 80/20 split, no gap filter). Beats cross-domain frozen probe 55.5% by +12.8pp. Also beats Humicroedit LoRA 64.8% — method generalises across datasets. Dataset had train/test leakage bug fixed before this run (prepare/hahackathon.py). |
| 2026-06-24 | Dataset cleanup: deleted discord/ and reward/ (abandoned experiments). Added humor_arena (2541 pairwise, 1213 usable wins) and humor_mechanics (120 jokes, ScaleAI-rated funniness 1–5) and reddit_jokes (83k rows, upvote-based). Created datasets/SOURCES.md registry. Fixed hahackathon 80/20 split. |
| 2026-06-25 | Investigated hri_tools (Baranov et al. 2023) dataset bundle — all 11 datasets are binary-labeled; no continuous crowd preference signal survives. reddit_jokes_last_laught (Weller & Seppi 2019) has 20k rows but binary labels and 2308 [removed] entries. Decided to keep our raw reddit_full_data.csv (upvote_ratio = continuous signal) instead. |
| 2026-06-25 | Reddit jokes prepare script (prepare/reddit_jokes.py): score >= 10 filter, 20–2000 char length, strip [removed]/[deleted], exact dedup → 5975 clean jokes saved to datasets/reddit_jokes/jokes.csv. upvote_ratio mean=0.859, range [0.55, 1.00]. Note: label range is [0.55, 1.0] not [0, 1] — will need per-dataset normalization before joint training. |
| 2026-06-25 | Pairwise training test (HaHa): in-batch pairwise BCE loss (28 pair signals per step of 8 jokes) vs regression MSE. Result: **67.7%** vs regression 68.3% — within ±2.2pp noise, effectively identical. Decision: use in-batch pairwise ranking loss for joint multi-dataset training. Eliminates cross-dataset label normalization problem; all datasets (rating or pairwise) convert to the same ranking signal. |
| 2026-06-25 | Downloaded HAHA Spanish Twitter 2019 (train + test_gold, 3.8MB). Investigated hri_tools/Oogiri-GO — deferred (on mobile data; also popularity bias concern since votes are visible). Open Mic marked as not publicly accessible. |
| 2026-06-25 | Joint training design decisions: (1) prompt must be dataset-specific — humicroedit needs "original headline: X / edited version: Y" context, not just the edited text; (2) HAHA Spanish non-humorous tweets (NaN funniness) treated as score=0 so any humorous tweet ranks above them in-batch; (3) pipeline order: separate LoRA per dataset → cosine similarity check → drop outliers → joint LoRA on compatible datasets. |
| 2026-06-25 | Individual LoRA cross-domain results (all trained with in-batch pairwise BCE, tested on HaHa pairwise n=2000, zero-shot baseline=55.4%): hahackathon 67.9% (+12.5pp, same domain), reddit_jokes 58.9% (+3.5pp), haha_spanish 57.0% (+1.6pp, within noise), humicroedit 54.8% (-0.6pp, within noise), humor_arena 52.8% (-2.6pp, near random). Cosine similarity between LoRA weight vectors all near zero (0.003–0.033 in 293M-dim space) — high-dimensionality makes all vectors orthogonal, metric uninformative. Abandoned cosine similarity compatibility check. |
| 2026-06-25 | Joint training launched on hahackathon + humicroedit + reddit_jokes + haha_spanish (51,817 examples, humor_arena excluded — binary 0/1 labels failed to learn useful signal). Dataset imbalance noted: haha_spanish 46% + humicroedit 35% = 81% of data. Balanced version (capped per dataset) flagged as follow-up experiment. Joint training currently running on 4090. |
| 2026-06-25 | TODO: write prepare/nycc.py and train NYCC LoRA (80/20 split) to validate NYCC as a meaningful benchmark — without this we cannot claim NYCC is a good eval set. TODO: balanced joint training with per-dataset cap as experiment 2. TODO: paper intro draft in paper/intro_notes.md. |
| 2026-06-26 | NYCC LoRA validated: trained on 2k pairs (4k rows) from fold0_train, tested on fold0_test+fold0_val (n=1020). NYCC pairwise acc 65.7% — clearly above zero-shot baseline (~55%), confirms NYCC is a learnable benchmark. HaHa acc 56.9% (near baseline) — NYCC LoRA domain-specific, doesn't transfer. Note: Hessel et al. human ceiling = 64.6%; we slightly exceed it with minimal training data. |
| 2026-06-26 | Joint LoRA trained on 5 datasets (hahackathon 3.9k + humicroedit 10k + reddit_jokes 10k + haha_spanish 10k + nycc 10k, cap=10k stratified random, humor_arena excluded). Results: HaHa pairwise 65.2% (n=2000), NYCC pairwise 63.2% (n=1020). Joint LoRA generalizes across both benchmarks without dataset-specific training — confirms multi-dataset crowd signal is transferable. Slightly below single-dataset LoRAs (hahackathon-only 67.7%, nycc-only 65.7%) as expected — trades domain fit for generalization. |
| 2026-06-26 | Paper narrative: LLM-as-judge fails for humor (54–57% zero-shot plateau). Alternatives: Goes et al. (2022) Crowd Score (LLM personas vote on jokes) — limited to 52 jokes, 4 personas, not validated at scale. Santurkar et al. (2023) show persona simulation doesn't faithfully reproduce crowd preferences. Our approach: supervised fine-tuning on crowd labels directly — closes the gap from 55% to 65%+. No need to reimplement Crowd Score; our zero-shot results already constitute the LLM-as-judge baseline. |
