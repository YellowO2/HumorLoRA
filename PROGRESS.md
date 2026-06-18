# Progress Log
**Goal**: Paper by ~2026-06-24 (flexible)

This file tracks experiments, results, and key references needed for writing the final paper.
Update the log section after each experiment.

---

## Paper Direction

We are building a collection of empirical findings about fine-tuning LLMs for humor preference alignment. Basically sort of what we discover along the way of trying to make LLMs humorous. A key question is what can be done to give LLM's sort of a taste of humor or even a good sense of humor.

The original mechanism hypothesis (self-preference bias → style shift → preference shift via De-GPT DPO) was explored but found weak: DPO produced no meaningful improvement on NYCC or SHP, and Discord SFT hurt performance across the board. The paper is now a findings paper — reporting what works, what doesn't, and why, across humor and general preference datasets.

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

5. **Gut vs no-gut prompt wording makes zero difference** across all models and datasets. Going forward: plain (no-gut) only.

---

## Datasets
| Name | Description | Split used | Size | Paper | Access |
|------|-------------|-----------|------|-------|--------|
| NYCC (New Yorker Caption Contest) | Funnier caption A or B? Crowd-aggregated ground truth | 5-fold validation | up to 2616 | Hessel et al. (2022) | https://arxiv.org/abs/2209.06293 |
| SHP (stanfordnlp/SHP) | Better Reddit comment A or B? Upvote-based preference, score_ratio ≥ 2 | validation | 2000 | Ethayarajh et al. (2022) | https://arxiv.org/abs/2110.08420 |
| HaHackathon (SemEval-2021 Task 7) | 1000 English jokes each rated 0–5 by multiple AMT crowd annotators | test set | 1000 jokes | Meaney et al. (2021) | https://aclanthology.org/2021.semeval-1.9 |
| Jester (dataset 3) | 150 English jokes rated -10 to +10 by 54,905 users; 140 usable; avg ratings -2.74 to +3.66 | full dataset | 140 usable jokes | Goldberg et al. (2001) | https://eigentaste.berkeley.edu/dataset/ |
| HAHA Spanish Twitter (2019) | Spanish tweets rated 0–4 for humor by screened crowd annotators (test tweets used to filter low-quality raters); 30k tweets | TBD | TBD — not yet downloaded | Chiruzzo et al. (2019) | https://www.fing.edu.uy/inco/grupos/pln/haha/ |
| Humicroedit / FunLines | ~15,000 English news headlines with word edits rated 0–3 for funniness by 5 AMT judges each; SemEval-2020 Task 7 | TBD | ~15,000 — not yet downloaded | Hossain et al. (2020) | https://arxiv.org/abs/2008.00304 |
| Open Mic Dataset | Stand-up comedy transcripts (~40 hrs); humor quotient derived from real audience laughter detection — behavioral ground truth, not annotation; validated vs 3 annotators (kappa=0.6) | TBD | TBD — not yet downloaded | Mittal et al. | https://arxiv.org/abs/2110.12765 |
| Oogiri-GO / Oogiri-Corpus | Japanese humor task; ~100 candidate responses per prompt each rated by ~100 independent blind judges — eliminates popularity bias | TBD | TBD — not yet downloaded | Murakami et al. / Zhong et al. | https://arxiv.org/abs/2512.21494 |
| Yelp Funny Reviews | Yelp reviews with crowd "funny" votes; general preference signal similar to SHP — useful if we expand beyond humor | TBD | Large — not yet sourced | de Oliveira & Rodrigo (Stanford CS224d) | No public repo |
| Cards Against Humanity (CAH Lab) | Real player gameplay choices — naturalistic behavioral ground truth, not paid annotation. **Not currently obtainable** — requires emailing lab | N/A | N/A | Cards Against LLMs | N/A — email lab |
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
| Meaney et al. (2021), *SemEval-2021 Task 7: HaHackathon, Detecting and Rating Humor and Offense* | https://aclanthology.org/2021.semeval-1.9 | HaHackathon dataset paper; English jokes rated 0–5 by crowd annotators (AMT); our eval uses 1000-joke test set, measured via Spearman r vs human avg rating |

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
