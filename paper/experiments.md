# Experiments

We evaluate a range of approaches to humor preference judgment on three pairwise benchmarks: HaHa pairwise (N=2000), NYCC pairwise (N=1020--2616 depending on the eval), and Humicroedit pairwise (N=2628). We also use SHP (Ethayarajh et al., 2022), a Reddit comment preference dataset, as a non-humor control to help distinguish humor preference gains from general capability changes. We use instruct instead of pretrained model unless stated otherwise. Greedy decoding throughout. 95% CI: ±2.2pp at n=2000, ±3.1pp at n=1020, ±1.9pp at n≈2600.

We organize the tested methods into two groups:

**Fail to improve over zero-shot (§5.1):**
- Prompt framing variants: gut-feeling, crowd-framing, CoT
- Style fine-tuning: Discord SFT (human conversation), De-GPT DPO (human-vs-AI)
- Persona simulation: Crowd Score, 4 HSQ humor archetypes (Goes et al., 2022)
- Humor basis prompting: 17-dimension decomposition (SemEval 2026)
- Few shot (5 examples)
- Varying Model size (2B--9B): no improvement trend across six base models

**Work (§5.2):**
- Linear activation probing on crowd-labeled funniness
- LoRA fine-tuning with pairwise preference objective, single-dataset and joint

---

## 5.1 What Fails

Table 1 summarizes all approaches that fail to meaningfully exceed zero-shot performance.

**Table 1: Approaches that do not improve over zero-shot**

| Approach | HaHa pairwise | NYCC pairwise | Humicroedit pairwise | Notes |
|---|---|---|---|---|
| Zero-shot (Gemma-4-E2B) | — | 52.3% | — | 2B model |
| Zero-shot (Qwen3-4B) | — | 52.0% | — | n=300 only |
| Zero-shot (Gemma-4-E4B) | — | 53.7% | — | |
| Zero-shot (Qwen3.5-4B) | 55.4% | 54.5% | 56.3% | Prompt framing (gut/plain/crowd) identical within noise |
| + CoT prompting (Qwen3.5-4B) | 54.4% | 53.2% | — | -1.0pp / -1.3pp, within noise |
| + Few-shot rating, 5 anchors (Qwen3.5-4B) | 56.3% | — | 57.1% | Derived pairwise from ratings; +0.9pp / +0.8pp, within noise |
| Zero-shot (Llama-3.1-8B) | — | ~50% | — | |
| Zero-shot (Hermes-3-8B) | — | 55.6% | — | |
| Zero-shot (Qwen3.5-9B) | — | 52.4% | — | 9B model; same plateau as 4B |
| Discord SFT (Hermes-3-8B) | — | 50.6% | — | -5pp vs base, general degradation |
| De-GPT DPO (Qwen3.5-4B) | 54.8% | 53.5% | — | Flat across all datasets |
| Crowd Score, Goes et al. (2022) (Qwen3.5-4B) | ~53% | — | — | 4 personas x same LLM; near random |
| 17-dim humor basis prompting (Qwen3.5-4B) | — | — | 56.1–56.3% | SemEval 2026 pointwise approach; no gain |
| Pairwise crowd probe (direct) (Qwen3.5-4B) | — | — | 56.7–60.4% | Probe on pairwise crowd signal; noisy labels undermine probe |

*† NYCC results in this table use varying eval sizes (n=1000–2616) across runs; all fall within the 53–56% plateau regardless of n.*

**Prompt framing variants.** We tested several prompt variants on Qwen3.5-4B: plain ("return A or B"), gut-feeling ("use your gut feeling"), crowd-framing ("which would a crowd find funnier"), and role-based ("you are an expert in humor"). All produce identical results within noise on both NYCC and HaHa pairwise (differences ≤1.0pp). We also tested few-shot rating prompts (5 labeled anchors from the training set), deriving pairwise predictions from the model's per-joke scores. This yields 56.3% on HaHa and 57.1% on Humicroedit, both within noise of zero-shot. Prompt variations appear to have no effect on humor judgment performance.

**Chain-of-thought prompting.** Chain-of-thought prompting produces a significant drop on SHP (-3.4pp, n=2000) and a neutral result on NYCC (-1.3pp, within CI). An explanation is that there is no clear reasoning path for subjective preference, hence verbalizing a rationale introduces noise rather than signal. This is consistent with Wang et al. (2025), who find CoT collapses LLM judgment distributions in 30/40 scoring scenarios.

**Varying model size.** Across six base models ranging from 2B to 9B parameters (Gemma-4-E2B 52.3%, Gemma-4-E4B 53.7%, Qwen3-4B 52.0%, Qwen3.5-4B 54.5%, Llama-3.1-8B ~50%, Hermes-3-8B 55.6%, Qwen3.5-9B 52.4%), all plateau within 52–56% on NYCC with no significant trend.

**Style fine-tuning.** The motivation was the self-preference bias hypothesis (Wataoka et al., 2024): LLMs prefer outputs stylistically similar to what they would generate. Since standard LLMs are RLHF-trained for helpfulness, they may favor helpful-sounding outputs even when judging humor. Fine-tuning on human conversation could shift the model's style away from assistant-speak, and in turn shift its humor preferences. However, DPO on human-vs-AI style data (De-GPT) is entirely flat across all benchmarks (within ±2.2pp noise), falsifying this hypothesis. SFT on Discord conversations degrades general capability (NYCC -5pp, HaHa r: 0.228→0.015, SHP -3.7pp), suggesting the style shift came at the cost of language understanding. The contrast with DPO (flat across all benchmarks) points to a capability issue rather than a failed preference shift.

**Crowd Score (Goes et al., 2022).** Crowd Score (Goes et al., 2022) achieves ~53% on HaHa pairwise, worse than zero-shot. The method attempts to simulate crowd judgment by prompting the same LLM four times with different humor personality instructions, but this fails to actually approximate a crowd's sense of humor. Santurkar et al. (2023) show LLMs have low persona steerability regardless of instruction, and the original evaluation covered only 52 jokes. Our evaluation at scale confirms the method does not generalize.

**17-dimension humor basis prompting.** We score each joke on 17 humor dimensions (Clear Punchline, Wordplay, Universality, etc.), following the SemEval 2026 lmfaoooo system, and aggregate into a pairwise prediction. This achieves 56.1–56.3% on Humicroedit pairwise, identical to zero-shot; a logistic regression on the 17-dim feature difference vector produces the same result, suggesting the features carry no useful signal without fine-tuning. Note that the full lmfaoooo system fine-tunes on these features, and their 17 humor dimensions were derived from their specific model's outputs rather than Qwen3.5-4B, so our results may not reflect the full method's potential.

---

## 5.2 What Works

**Table 2: Our approaches on held-out benchmarks** *(Qwen3.5-4B)*

| Approach | HaHa pairwise | NYCC pairwise | Humicroedit pairwise | Notes |
|---|---|---|---|---|
| Zero-shot baseline | 55.4% | 54.5% | 56.3% | For reference |
| Individual human agreement | — | 64.6% | — | NYAcc (Hessel et al., 2023) |
| SemEval 2020 winner (Hitachi) | — | — | 67.43% | Fine-tuned ensemble on Humicroedit |
| Activation probe (binary labels) | — | — | 61.9% | Layer 16; trained on Humicroedit; cross-domain HaHa transfer: 55.5% (near baseline) |
| Margin-filtered probe (score gap ≥1.0) | — | — | 67.0% | High-confidence pairs only (n=204 test) |
| LoRA regression (single dataset) | 68.3% | 65.7% | 64.8% | MSE on crowd ratings |
| Pairwise LoRA (single dataset) | 67.7% | 65.7% | — | In-batch BCE; ≈ regression within noise |
| **Joint pairwise LoRA (5 datasets)** | 65.2% | 63.2% | 59.8% | Our main result used for demo |


**Activation probing.** We extract hidden-state activations using the prompt template from Zou et al. (2023): *"Consider the amount of funniness in the following: [text]"*, then train a logistic regression probe on binary funny/unfunny labels from Humicroedit. The probe achieves 61.9% on Humicroedit pairwise (layer 16). Cross-domain transfer is limited (55.5% on HaHa), indicating the direction is partly domain-specific. Filtering to high-margin pairs (score gap ≥1.0) yields 67.0%, matching the SemEval 2020 winner on clean signal.

**Single-dataset LoRA.** We fine-tune a LoRA adapter (Hu et al., 2021) using the same prompt template, *"Consider the amount of funniness in the following: [text]"*, to score each joke, then apply MSE loss against crowd ratings (regression variant) or in-batch pairwise BCE loss (pairwise variant). Regression achieves 68.3% on HaHa pairwise and 65.7% on NYCC, the latter matching human agreement (64.6%) using only 2,000 training pairs. Pairwise BCE produces equivalent results (67.7% HaHa), validating it as the training objective for joint training. We also test each single-dataset LoRA on HaHa to measure cross-domain transfer (zero-shot baseline: 55.4%): reddit\_jokes achieves 58.9% (+3.5pp), haha\_spanish 57.0% (+1.6pp, within noise), and humicroedit 54.8% (no transfer).

**Joint pairwise LoRA.** The motivation is that training on diverse humor domains, rather than a single domain, should produce a more general sense of humor. Training jointly on all five datasets (hahackathon, humicroedit, reddit_jokes, haha_spanish, nycc; capped at 10k rows each, within-source masking) achieves 65.2% on HaHa pairwise, 63.2% on NYCC, and 59.8% on Humicroedit. There is a slight drop versus single-dataset LoRAs, but is expected due to less overfitting. Training on all five datasets does not significantly dilute per-domain performance, suggesting they share a learnable common direction. The larger drop on Humicroedit (-5pp from its single-dataset 64.8%) is likely due to the training cap limiting Humicroedit-specific exposure, and news headline edits being a structurally different domain from the other four text-joke-style datasets.
