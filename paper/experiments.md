# Experiments

We evaluate a range of approaches to humor preference judgment on three pairwise benchmarks: **HaHa pairwise** (N=2000), **NYCC pairwise** (N=1020 for LoRA eval, N=2616 otherwise), and **Humicroedit pairwise** (N=2628). All experiments use Qwen3.5-4B as the base model unless stated otherwise. Greedy decoding (do_sample=False) throughout. 95% CI: ±2.2pp at n=2000, ±3.1pp at n=1020, ±1.9pp at n≈2600.

We organize methods into two groups:

**Fail to improve over zero-shot (§5.1):**
- Prompt framing variants: gut-feeling, crowd-framing, CoT
- Style fine-tuning: Discord SFT (human conversation), De-GPT DPO (human-vs-AI)
- Persona simulation: Crowd Score, 4 HSQ humor archetypes (Goes et al., 2022)
- Humor basis prompting: 17-dimension decomposition (SemEval 2026)

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

**Prompt framing variants.** We tested several prompt variants on Qwen3.5-4B: plain ("return A or B"), gut-feeling ("use your gut feeling"), crowd-framing ("which would a crowd find funnier"), and role-based ("you are a humor expert"). All produce identical results within noise on both NYCC and HaHa pairwise (differences ≤1.0pp). We also tested few-shot rating prompts (5 labeled anchors from the training set), deriving pairwise predictions from the model's per-joke scores. This yields 56.3% on HaHa and 57.1% on Humicroedit, both within noise of zero-shot. The humor judgment ceiling appears to be in the model weights, not the prompt.

**Varying Model size.** Across six base models ranging from 2B to 9B parameters (Gemma-4-E2B 52.3%, Gemma-4-E4B 53.7%, Qwen3-4B 52.0%, Qwen3.5-4B 54.5%, Llama-3.1-8B ~50%, Hermes-3-8B 55.6%, Qwen3.5-9B 52.4%), all plateau within 52–56% on NYCC with no upward trend.

**Chain-of-thought prompting.** Chain-of-thought prompting produces a significant drop on SHP (-3.4pp, n=2000) and a neutral result on NYCC (-1.3pp, within CI). An explanation is that there is no clear reasoning path for subjective preference, hence verbalizing a rationale introduces noise rather than signal. This is consistent with Wang et al. (2025), who find CoT collapses LLM judgment distributions in 30/40 scoring scenarios.

**Style fine-tuning.** Our motivation was the self-preference bias hypothesis (Wataoka et al., 2024): LLMs prefer outputs stylistically similar to themselves, so fine-tuning on human conversation might shift the model's style toward human-majority text, indirectly shifting its humor preferences. DPO on human-vs-AI style data (De-GPT) is entirely flat across all benchmarks (within ±2.2pp noise), falsifying this hypothesis. SFT on Discord conversations goes further: it actively degrades general capability (NYCC -5pp, HaHa r: 0.228→0.015, SHP -3.7pp), likely because aggressive SFT on casual conversation damages language understanding, which then hurts humor judgment downstream. The contrast between DPO (flat) and SFT (harmful) suggests the degradation is a capability issue, not a failed preference shift.

**Crowd Score (Goes et al., 2022).** Crowd Score (Goes et al., 2022) achieves ~53% on HaHa pairwise, worse than the zero-shot baseline. Calling the same LLM four times with different humor personality instructions produces nearly identical outputs. This empirically confirms the theoretical critique: HSQ personality types describe how people habitually use humor in their lives, not dimensions along which jokes vary; and persona steerability in LLMs is low (Santurkar et al., 2023).

**17-dimension humor basis prompting.** Inspired by the SemEval 2026 lmfaoooo system, we score each joke on 17 humor dimensions (Clear Punchline, Wordplay, Universality, etc.) and aggregate into a pairwise prediction. This pointwise variant achieves 56.1–56.3% on Humicroedit pairwise, identical to zero-shot. The model assigns nearly identical feature scores to most jokes regardless of funniness, suggesting pointwise dimension scoring collapses without further supervision. Note that the full lmfaoooo system additionally fine-tunes on these features, and their 17 dimensions were derived empirically from their specific LLM's outputs — not Qwen3.5-4B. We only evaluate the prompting component with their published feature set, so results may not reflect the full method's potential.

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
| LoRA regression (single dataset) | 68.3% | 65.7%† | 64.8% | MSE on crowd ratings |
| Pairwise LoRA (single dataset) | 67.7% | 65.7%† | — | In-batch BCE; ≈ regression within noise |
| **Joint pairwise LoRA (5 datasets)** | **65.2%** | **63.2%** | [TODO] | Our main result; Humicroedit eval pending (training source — partial in-domain) |

†Single-dataset NYCC LoRA (2k training pairs).

**Activation probing.** We extract hidden-state activations using the prompt template from Zou et al. (2023): *"Consider the amount of funniness in the following: [text]"*, then train a logistic regression probe on binary funny/unfunny labels from Humicroedit. The probe achieves 61.9% on Humicroedit pairwise (layer 16). Cross-domain transfer is limited (55.5% on HaHa), indicating the direction is partly domain-specific. Filtering to high-margin pairs (score gap ≥1.0) yields 67.0%, matching the SemEval 2020 winner on clean signal.

**Single-dataset LoRA.** We fine-tune a LoRA adapter (Hu et al., 2021) using the same prompt template, *"Consider the amount of funniness in the following: [text]"*, to score each joke, then apply MSE loss against crowd ratings (regression variant) or in-batch pairwise BCE loss (pairwise variant). Regression achieves 68.3% on HaHa pairwise and 65.7% on NYCC, the latter slightly above the human ceiling of 64.6% using only 2,000 training pairs. Pairwise BCE produces equivalent results (67.7% HaHa), validating it as the training objective for joint training.

**Individual LoRA cross-domain transfer.** We test each single-dataset pairwise LoRA on HaHa (zero-shot baseline: 55.4%) to measure how much humor signal transfers across domains. reddit\_jokes LoRA achieves 58.9% (+3.5pp), haha\_spanish 57.0% (+1.6pp, within noise), and humicroedit 54.8% (no transfer). The pattern is domain-dependent: text joke datasets (reddit\_jokes) partially transfer to HaHa text jokes, while news headline edits (humicroedit) do not.

**Joint pairwise LoRA.** Training jointly on all five datasets (hahackathon, humicroedit, reddit\_jokes, haha\_spanish, nycc; capped at 10k rows each, within-source masking) achieves 65.2% on HaHa pairwise and 63.2% on NYCC. This is a slight drop versus single-dataset LoRAs but a +10pp gain over zero-shot. The crowd preference signals from diverse humor domains are compatible: adding four other datasets does not significantly dilute per-domain performance, suggesting they share a learnable common direction.
