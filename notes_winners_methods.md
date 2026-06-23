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

## Open questions
- Which backbone to use for Step 1? BERT-base is closest to Hitachi; qwen3.5:4b is our existing setup
- Scale normalization for combined training: Humicroedit 0–3 vs HaHa 0–5 (just divide/multiply by 5/3?)
- NYCC: only has pairwise votes, no continuous rating — need Elo or skip it
