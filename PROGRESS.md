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

2. **De-GPT DPO has no effect** on either NYCC or SHP — results within ±2.2pp noise. Perplexity shift from DPO was never measured (open gap), but the behavioural signal is negative.

3. **All base models plateau at ~53–56% on NYCC** regardless of architecture (Gemma4, Qwen3.5, Hermes-3). Human ceiling is ~64.6% (NYAcc). The gap is large and no fine-tuning approach tested has closed it.

4. **CoT prompting hurts SHP accuracy by ~3.5pp** (significant at n=2000, CI ±2.2pp) across both models tested. The effect is not detectable on NYCC at n=1000 (CI ±3.1pp — underpowered). Note: "thinking" here is prompt-level CoT (model asked to explain before answering), not built-in reasoning tokens. Consistent with Wang et al. (2025).
   - **Theory**: Humor preference may be intuition-driven. Crowd humor signal has no reasoning path to anchor to — there is no "correct" logic for why something is funny. By contrast, SHP (Reddit upvotes) has a more structured signal (helpfulness, engagement) that CoT can reason toward or away from. This would predict CoT genuinely doesn't hurt humor judgment even at n=2000 — pending experiment.

5. **Gut vs no-gut prompt wording makes zero difference** across all models and datasets. Going forward: plain (no-gut) only.

---

## Datasets
| Name | Task | Split used | Size |
|------|------|-----------|------|
| NYCC (New Yorker Caption Contest) | Funnier caption A or B? | 5-fold validation | up to 2616 |
| SHP (stanfordnlp/SHP) | Better Reddit comment A or B? | validation, score_ratio ≥ 2 | 2000 |
| HaHackathon | Joke rating 0–5, Spearman r vs human avg | test | varies |
| Discord-Dialogues (`mookiezi/Discord-Dialogues`) | SFT training — casual human multi-turn chat | first 50,000 | 50,000 |
| De-GPT-DPO (`qingy2024/De-GPT-DPO`) | DPO training — human (chosen) vs AI (rejected) | first 5,000 | 5,000 |

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
| NYCC | qwen3.5:4b | 54.5% | 55.0% (n=1000) | +0.5pp (underpowered) |
| NYCC | qwen4b-degpt-dpo | 53.5% | 52.2% (n=1000) | -1.3pp (underpowered) |
| SHP | qwen3.5:4b | 63.2% | 59.8% (n=2000) | **-3.4pp** |
| SHP | qwen4b-degpt-dpo | 63.8% | 59.7% (n=2000) | **-3.7pp** |

NYCC CoT results at n=1000 are underpowered for detecting a 3.5pp effect. Pending: extend qwen3.5:4b CoT to n=2000 on NYCC.

---

## Results
| Model | Training | NYCC (plain) | SHP (plain) | HaHa (Spearman r) |
|-------|----------|--------------|-------------|-------------------|
| gemma4:e4b | base | 53.7% | — | — |
| gemma4-e4b-discord | SFT (discord) | 50.6% | — | — |
| qwen3.5:4b | base | 54.5% (n=2616) | 63.2% (n=2000) | — |
| qwen4b-degpt-dpo | DPO (De-GPT) | 53.5% (n=1000) | 63.8% (n=2000) | — |
| hermes-3-8b | base (SFT+DPO synthetic) | 55.6% | 59.5% (n=2000) | 0.228 (n=671) |
| discord-hermes-3-8b | + Discord SFT | 50.6% | 55.8% (n=2000) | 0.015 (n=1000) |
| llama-3.1-8b-instruct | RLHF-aligned | ~50% | — | 0.277 (n=971) |

---

## TODO
1. **Run qwen3.5:4b CoT on NYCC n=2000** (extend existing n=1000 by running examples 1000–1999, then merge) — confirms or denies the humor-is-intuitive theory
2. ~~**Run discord-hermes-3-8b on SHP**~~ ✓ Done — hermes 59.5% vs discord-hermes 55.8%, degradation is general
3. **Decide paper scope** — current findings support a "what doesn't work and why" framing; decide whether to add a positive result (e.g. reward model / Option A pivot)

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
