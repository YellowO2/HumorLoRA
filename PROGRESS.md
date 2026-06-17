# Progress Log
**Goal**: Paper in ~10 days (by ~2026-06-22)

This file tracks experiments, results, and key references needed for writing the final paper.
Update the log section after each experiment.

---

## Goal
Make a model's preference judgments align with what the **majority of humans would prefer** — not any single person's taste, but the aggregate crowd signal (as measured by datasets like NYCC and SHP).

## Hypothesis
The mechanism to achieve this:
1. Prior research shows LLM-as-a-Judge exhibits self-preference bias — it favours outputs stylistically similar to itself.
2. Instruction-tuned models are optimised for helpfulness and compliance, not human-like expression. Their generation style is "instruct-style", not how humans naturally talk.
3. Because of self-preference bias, instruct-tuned judges likely favour instruct-style outputs — which misaligns with what the majority of humans prefer.
4. Therefore: if we fine-tune the model on natural human conversation (De-GPT DPO on Reddit/Discord), its generation style shifts toward human-like text → its preference judgments should also shift toward majority human preference.

**In short**: fix the style gap → fix the preference gap, via self-preference bias as the linking mechanism.

**Grounded in**:
- Wataoka et al. (2024), *Self-Preference Bias in LLM-as-a-Judge*, ICLR 2025 submission
  https://arxiv.org/pdf/2410.21819
  → Self-preference bias stems from perplexity familiarity: models prefer outputs that look like what they would generate themselves.

---

## Datasets
| Name | Task | Split used | Size |
|------|------|-----------|------|
| NYCC (New Yorker Caption Contest) | Funnier caption A or B? | 5-fold validation | 2000 |
| SHP (stanfordnlp/SHP) | Better Reddit comment A or B? | validation, score_ratio ≥ 2 | 2000 |
| Discord-Dialogues (`mookiezi/Discord-Dialogues`) | SFT style training — casual human multi-turn conversation | first 50,000 examples | 50,000 |

Human baselines (from Hessel et al. 2022):
- NYCC CrowdAcc: 83.7% (humans predicting crowd preference)
- NYCC NYAcc: 64.6% (humans predicting editor choice)
- Our eval is ~71% official_winner / ~29% crowd_winner → relevant human ceiling is ~64.6%

SHP: no inter-annotator agreement reported — preferences inferred from Reddit upvotes, not explicit human labels. score_ratio ≥ 2 filter used for cleaner signal.

---

## Eval prompt modes
Three modes were tested:
- **gut**: `"Use your gut feeling and return <answer>A</answer> or <answer>B</answer>."`
- **no-gut** (plain): `"Return <answer>A</answer> or <answer>B</answer>."`
- **thinking**: `"Briefly explain why each is good or bad, then return your final choice as <answer>A</answer> or <answer>B</answer>."` + model's built-in thinking enabled

**Finding**: gut vs no-gut makes zero difference across all models and datasets. Going forward, use **no-gut (plain)** only — no need to run both.

Thinking mode results:

| Dataset | Model | plain | thinking | delta |
|---------|-------|-------|----------|-------|
| NYCC | qwen3.5:4b | 54.5% | 55.0% | +0.5pp (noise) |
| NYCC | qwen4b-degpt-dpo | 53.5% | 52.2% | -1.3pp (noise) |
| SHP | qwen3.5:4b | 63.2% | 59.8% | **-3.4pp** |
| SHP | qwen4b-degpt-dpo | 63.8% | 59.7% | **-3.7pp** |

Thinking hurts SHP (~3.5pp) but not NYCC. Likely because NYCC is already near chance (~55%) — there's no signal for thinking to degrade. SHP has a real signal at 63% and thinking mode erodes it. Claim: **thinking mode hurts preference judgment when there is a real signal to degrade; on near-chance tasks the effect is invisible.**

---

## Results
| Model | Training | NYCC (plain) | SHP (plain) | HaHa (Spearman r) |
|-------|----------|--------------|-------------|-------------------|
| gemma4:e4b | base | 53.7% | — | — |
| gemma4-e4b-discord | SFT (discord) | 50.6% | — | — |
| qwen3.5:4b | base | 54.5% (n=2616) | 63.2% (n=2000) | — |
| qwen4b-degpt-dpo | DPO (De-GPT) | 53.5% (n=1000) | 63.8% (n=2000) | — |
| hermes-3-8b | base (SFT+DPO synthetic) | 55.6% | — | 0.228 (n=671) |
| discord-hermes-3-8b | + Discord SFT | 50.6% | — | 0.015 (n=1000) |
| llama-3.1-8b-instruct | RLHF-aligned | ~50% | — | 0.277 (n=971) |

- Discord SFT consistently hurts NYCC accuracy across two different base models: -3pp on Gemma4, -5pp on Hermes-3. Pattern is robust.
- **Discord SFT destroys HaHackathon humor correlation**: base Hermes r=0.228 → Discord-Hermes r=0.015 (near random). Same pattern holds across two datasets and two eval methods.
- DPO on De-GPT made no meaningful improvement on either NYCC or SHP (within ±2.2pp noise).
- Base models (Gemma4, Qwen3.5, Hermes-3) all converge to ~53-56% on NYCC regardless of architecture.
- Eval is fully deterministic (do_sample=False, greedy decoding) — re-running on the same examples gives identical results.
- 95% CI at n=2000 is ±2.2pp (SE = √(0.55×0.45/2000) ≈ 0.011). The ~2pp gap between base and DPO is within noise — not statistically significant.
- All models have weak humor correlation (r=0.22–0.28) on HaHackathon — expected, humor is genuinely subjective. Discord SFT is the outlier at r=0.015.

---

## Key Questions / Gaps
1. Did DPO actually shift the model's perplexity distribution? (never measured)
   - If yes but accuracy didn't improve → perplexity shift doesn't drive humor preference
   - If no → training was insufficient
2. Is NYCC near ceiling regardless of method? (~55% vs human ~60-65%)

---

## References
| Paper | Link | Key takeaway for this paper |
|-------|------|-----------------------------|
| Wataoka et al. (2024), *Self-Preference Bias in LLM-as-a-Judge* | https://arxiv.org/abs/2410.21819 | Self-preference bias stems from perplexity familiarity — models prefer outputs stylistically similar to themselves |
| Verga et al. (2024), *A Survey on LLM-as-a-Judge* | https://arxiv.org/abs/2411.15594 | Documents known biases: self-enhancement, verbosity, position; GPT-4 only agrees with humans ~80% of the time; alignment degrades on subjective/cultural preferences like humor |
| Zheng et al. (2023), *Chatbot Arena* | https://arxiv.org/abs/2403.04132 | Crowdsourced human preference benchmark — the gold standard for real human preference data |
| Hessel et al. (2022), *Do Androids Laugh at Electric Sheep?* | https://arxiv.org/abs/2209.06293 | NYCC dataset paper; human CrowdAcc=83.7%, NYAcc=64.6%; our eval is ~71% editor labels so ceiling ≈ 64.6% |
| Ethayarajh et al. (2022), *Understanding Dataset Difficulty with V-Usable Information* | https://arxiv.org/abs/2110.08420 | SHP dataset paper (ICML 2022 Outstanding Paper); SHP preferences inferred from Reddit upvotes, no IAA reported |
| Kirk et al. (2024), *The PRISM Alignment Dataset* | https://arxiv.org/abs/2404.16019 | NeurIPS 2024; 1,500 participants from 75 countries rating 21 LLMs on subjective/value-laden topics — directly shows human preferences are diverse, individualized, and cross-culturally inconsistent; supports why collapsing "human taste" into a single signal is hard |
| Wang et al. (2025), *Improving LLM-as-a-Judge Inference with the Judgment Distribution* | https://arxiv.org/abs/2503.03064 | EMNLP 2025 Findings; CoT collapses the judgment distribution and hurts LLM-as-a-judge in 30/40 scoring cases. Corroborates our SHP finding that thinking mode degrades preference accuracy (~3.5pp drop). Null result on NYCC likely due to low power at n=1000 (CI ±3.1pp), not a true floor effect. |

---

## Training
- **DPO**: `qingy2024/De-GPT-DPO` — human (chosen) vs AI (rejected), 5000 examples, 1 epoch
  - Checkpoint: `outputs/qwen4b-degpt-dpo/checkpoint-625`
  - Token lengths verified: prompt avg 18 tokens, chosen avg 104 — no truncation issues

---

## TODO (priority order)
1. **Decide paper direction** — original hypothesis (style shift → preference shift) has weak evidence and a confound (Discord SFT degradation may be general reasoning degradation, not humor-specific). Two options:
   - **Option A (pivot)**: humor reward model that generalises across domains — train on NYCC, test transfer to HaHa/CAH/Spanish Twitter
   - **Option B (salvage)**: run SHP on discord-hermes-3-8b to check if SFT degradation is general or humor-specific; run Mistral DPO experiment to close the RLHF-resistance loophole
2. **If Option A**: collect CAH and Spanish Twitter datasets, design reward model training pipeline
3. **If Option B**: run discord-hermes SHP eval (fast, ~20 min), then decide if Mistral DPO is worth it

---

## Log
| Date | Action |
|------|--------|
| 2026-06-09 | Ran base qwen3.5:4b on SHP (all 3 prompt variants). Fixed summary.csv path in run_eval_shp.py. |
| 2026-06-12 | Confirmed negative result. Verified DPO training bug was not an issue. Identified perplexity gap as missing experiment. |
| 2026-06-13 | HaHackathon rating eval: Hermes r=0.228, Discord-Hermes r=0.015, Llama r=0.277. Discord SFT destroys humor correlation — pattern confirmed across NYCC + HaHa. |
| 2026-06-17 | Confirmed gut vs no-gut prompt makes zero difference. Thinking mode hurts SHP by ~3.5pp but not NYCC (already near chance). Going forward: plain (no-gut) only. Updated results table with exact numbers. Reconsidering paper direction — original hypothesis has confound and mechanism issues. |
