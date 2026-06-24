# SemEval Winners: Methods Analysis

## Goal
Understand what the winners actually did so we can replicate and extend.
Verdict: both use the same core recipe — **supervised regression on crowd labels + ensemble**.

---

## Hitachi — SemEval 2020 Task 7 (Humicroedit)
**Result**: 67.43% pairwise accuracy (Task 1c)  
**Paper**: [ACL Anthology](https://aclanthology.org/2020.semeval-1.101/)  
**Team**: Terufumi Morishita et al.

### Method: Stacking at Scale (SaS)
1. PLMs used: **BERT, GPT-2, RoBERTa, XLNet, Transformer-XL, XLM** (6–7 model families)
2. For each PLM: fine-tune 50 instances with unique hyperparameter settings
3. 5-fold cross validation → select 20 best per PLM
4. Total: **700 models** (7 PLMs × 20 settings × 5 folds)
5. Combine predictions via **Ridge regression** (not just averaging)
6. For subtask-2 pairwise: feed both headlines through scorer → pick higher score

### Key details (corrected)
- Input: **sentence pair format** — original headline + replacement word marked with special tokens (NOT just the edited headline)
- Uses additional **FunLines** training data on top of subtask-1
- Label: meanGrade from subtask-1 (0–3 continuous)
- Loss: MSE regression
- Ensemble: Ridge regression meta-learner on 700 base models' predictions
- No multi-task — humor only

### Runner-up systems (for reference)
- **Amobee (2nd)**: BERT + RoBERTa + XLNet, 30 instances each = 90 models ensemble
- **YNU-HPCC (3rd)**: edited headline only (no pair format), FastText/W2V/ELMo/BERT encodings → BiGRU → XGBoost. No PLM fine-tuning.
- **MLEngineer (4th)**: 4 BERT regression models + RoBERTa + Naive Bayes, edited headline only

---

## abcbpc — SemEval 2021 Task 7 HaHackathon
**Result**: RMSE 0.4959 on subtask 1b (humor rating, 0–5), #1 overall  
**Paper**: [ACL Anthology](https://aclanthology.org/2021.semeval-1.35/)  
**Models**: ERNIE 2.0 + DeBERTa

### Method: Multi-task + Ensemble
1. Use **ERNIE 2.0** and **DeBERTa** as backbone models
2. **Multi-task training**: jointly predict humor rating AND offense rating simultaneously (shared representation)
3. Fine-tune on crowd-annotated HaHa training data
4. **Ensemble**: combine predictions from ERNIE and DeBERTa (and variants)
5. Subtask 1a (is_humor binary): derived from regression score threshold

### Key details
- Input: joke text only
- Labels: humor_rating (0–5) + is_offensive (0–5) — multi-task
- Multi-task helps because humor/offense share linguistic features
- ERNIE 2.0: Chinese pre-trained model with knowledge-enhanced pretraining (still works on English)

---

## Comparison

| | Hitachi 2020 | abcbpc 2021 |
|---|---|---|
| Dataset | Humicroedit (0–3) | HaHa (0–5) |
| Backbone | BERT/RoBERTa/GPT-2 etc. | ERNIE 2.0 + DeBERTa |
| Training | Regression on crowd labels | Multi-task regression (humor + offense) |
| Ensemble | Stacking (meta-learner) | Average/weighted of ERNIE+DeBERTa |
| Pairwise | Derived from score | Derived from score |
| Key differentiator | Scale of ensemble | Multi-task signal |

**Core method is identical**: supervised regression on crowd labels → derived pairwise. Differences are in backbone choice and ensemble strategy.

---

## Our Plan

### Step 1 — Replicate Hitachi on Humicroedit (single model, no ensemble)
- Fine-tune one model (qwen3.5:4b or a BERT-family model) on subtask-1 training data
- Regression: predict meanGrade (0–3) from edited headline text
- Eval: derive pairwise accuracy on subtask-2 test
- Target: beat our zero-shot 56.3%; see how close to 67.43% with just one model

### Step 2 — Cross-dataset eval (zero fine-tuning)
- Take the Humicroedit-trained scorer
- Run on HaHa pairwise (compare predicted humor_rating → pairwise accuracy)
- This tells us: does supervised humor preference generalize across datasets?

### Step 3 — Combined training (if cross-eval is weak)
- Train on Humicroedit subtask-1 + HaHa rating data combined (normalize scales)
- Re-eval on both datasets
- Optionally add NYCC pairwise (convert to implicit rating via Elo or similar)

### Step 4 — Demo pipeline
- Use trained scorer as reward signal
- LLM generates funny responses; scorer ranks/filters them
- Conversational agent persona for group chat

---

---

## PAI — SemEval 2025 Task 11 (Emotion Detection, subjective task)
**Result**: 1st place in 19/28 languages (Track A) and 10/11 languages (Track B)  
**Paper**: [ACL Anthology](https://aclanthology.org/2025.semeval-1.150/)  
**Overview**: [arXiv 2503.07269](https://arxiv.org/abs/2503.07269)

### Why relevant
Task 11 is emotion intensity detection — crowd-annotated subjective labels, pairwise preference structure, same problem type as humor. Shows how the field moved from BERT (2020–21) to modern LLMs (2025).

### Method: 3-layer pipeline

#### Layer 1 — Base models (5 LLMs)
- ChatGPT-4o, DeepSeek-V3: used **zero/few-shot only** (no fine-tuning)
- Gemma-9b-it, Qwen-2.5-32b-instruct, Mistral-Small-24B: fine-tuned as **embedding models**

#### Layer 2 — "LLM as embedding model" (KEY IDEA)
Instead of generating text output, they:
1. Prompt: `"Detect the emotion of this sentence: {text}"`
2. Run through LLM transformer layers
3. **Mean-pool final-layer hidden states** → sentence embedding vector
4. Add **regression/classification head** on top
5. Fine-tune backbone with **AdaLoRA** (lr=1e-5, 10 epochs, 5-fold CV)

This is architecturally identical to our activation probe — extract hidden states, train a head — except they also fine-tune the backbone with AdaLoRA.

#### Layer 3 — Two-round ensemble
- Round 1: feed all 5 models into 4 meta-learners (3-layer NN, XGBoost, LightGBM, polynomial regression)
- Round 2: weighted voting, weight = dev-set Pearson × Jensen-Shannon divergence correction
- Ensemble gain: ~0.01–0.02 over best single model — small but consistent, separated 1st from 2nd

#### Prompt optimization (for zero-shot models)
Iterative loop up to 10 iterations:
1. Generate candidate prompts via "ContextAugment" (inject labeled examples) and "StructVar" (LLM paraphrases)
2. Prune low performers on dev set
3. Keep top-5 prompts + 2-3 few-shot examples at inference

### Key finding from overview paper
> "LLM-based approaches significantly outperformed traditional BERT models overall. Instruction fine-tuning using LoRA in combination with prompt design and data augmentation proved most effective."

BERT-family models (DeBERTa, mBERT, XLM-R) still competitive but clearly second-tier compared to LoRA-fine-tuned LLMs.

### Comparison to our probe

| | Our probe | PAI method |
|---|---|---|
| Backbone | Qwen (frozen) | Gemma/Qwen/Mistral (AdaLoRA fine-tuned) |
| Hidden states | Specific layer, last token | Mean-pool final layer |
| Head | Logistic regression | Regression/classification head |
| Supervision | Binary funny/unfunny | Crowd intensity labels |
| Backbone updates | None | Yes (AdaLoRA) |

**Our natural upgrade**: add AdaLoRA to Qwen + train regression head jointly on Humicroedit crowd labels → should push from 61.9% toward 67%+.

---

## BERT basics (2020–21 era) — How it works

**Pretraining**: BERT trained on huge text corpora with masked language modeling — randomly hide 15% of words, predict them. Forces the model to build deep contextual representations. Result: rich general-purpose text encoder.

**Fine-tuning for a task**: 
1. Take pretrained BERT
2. Add one linear layer on top (the "head"): hidden_state → score
3. Feed labeled examples: `[CLS] text [SEP]` → crowd label
4. Backpropagate through the entire model — all weights update slightly
5. After fine-tuning, the model has incorporated the crowd signal

**Why this beats zero-shot**: zero-shot LLMs have no exposure to what *this specific crowd* finds funny. Fine-tuning literally bakes in the crowd's taste.

**Sentence pair format** (used by Hitachi): `[CLS] original headline [SEP] edited headline [SEP]` — BERT's [CLS] token aggregates cross-attention over both, capturing the *relationship* (what changed) rather than just the edited text alone.

---

## Evolution summary: 2020 → 2021 → 2025

| Year | Task | Method | Result |
|------|------|--------|--------|
| 2020 | Humicroedit pairwise | BERT/RoBERTa/GPT-2 fine-tune × 700 ensemble | 67.43% |
| 2021 | HaHa rating | ERNIE + DeBERTa multi-task fine-tune | RMSE 0.4959 |
| 2025 | Emotion intensity | GPT-4o + DeepSeek + Qwen-32b LoRA + ensemble | 1st in 19/28 lang |

Core recipe unchanged: supervised on crowd labels + ensemble. Only the backbone upgraded.

---

## Open questions
- Which backbone to use for Step 1? BERT-base is closest to Hitachi; qwen3.5:4b is our existing setup
- Scale normalization for combined training: Humicroedit 0–3 vs HaHa 0–5 (just divide/multiply by 5/3?)
- NYCC: only has pairwise votes, no continuous rating — need Elo or skip it
