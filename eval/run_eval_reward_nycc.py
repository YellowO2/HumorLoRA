# ═══════════════════════════════════════════════════════════════════════════════
# Eval: humor reward model on NYCC pairwise (generalization test)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Trained on: HaHa + Jester pairwise (jokes)
# Tested on:  NYCC validation folds (caption contest — unseen domain)
#
# For each pair: score caption_a and caption_b independently,
# predict whichever has higher reward score, compare to crowd/editor label.
#
# Run: python eval/run_eval_reward_nycc.py
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

CHECKPOINT  = Path(__file__).parent.parent / "outputs" / "qwen4b-humor-reward" / "best"
BASE_MODEL  = "unsloth/Qwen3.5-4B"
DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "newyorker"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "reward"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"
MAX_LENGTH  = 256


# ── Load model ────────────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    quantization_config=bnb_config,
    device_map="auto",
)
base.config.pad_token_id = tokenizer.pad_token_id

print(f"Loading LoRA adapter from {CHECKPOINT}...")
model = PeftModel.from_pretrained(base, str(CHECKPOINT))
model.eval()
print("Model ready.")


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.inference_mode()
def score(texts: list[str]) -> list[float]:
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    out = model(**enc)
    logits = out.logits
    # handle unexpected shape — take first column if >1 output
    if logits.dim() > 1 and logits.shape[-1] > 1:
        logits = logits[:, 0]
    else:
        logits = logits.squeeze(-1)
    return logits.float().cpu().tolist()


# ── Load NYCC ─────────────────────────────────────────────────────────────────

dfs = []
for fold in range(5):
    df = pd.read_csv(DATASETS_DIR / f"fold{fold}_validation.csv")
    df["fold"] = fold
    dfs.append(df)
examples = pd.concat(dfs, ignore_index=True)
print(f"\nLoaded {len(examples)} NYCC pairs across 5 folds")


# ── Eval ──────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = RESULTS_DIR / f"reward_nycc_{timestamp}.csv"

results = []
BATCH = 16

for start in range(0, len(examples), BATCH):
    batch = examples.iloc[start:start + BATCH]
    captions_a = batch["caption_a"].tolist()
    captions_b = batch["caption_b"].tolist()

    scores_a = score(captions_a)
    scores_b = score(captions_b)

    for i, row in enumerate(batch.itertuples()):
        sa, sb = scores_a[i], scores_b[i]
        prediction = "A" if sa > sb else "B"
        expected   = str(row.expected).strip().upper()
        is_correct = prediction == expected

        results.append({
            "sample_id":   row.sample_id,
            "fold":        row.fold,
            "expected":    expected,
            "prediction":  prediction,
            "score_a":     round(sa, 4),
            "score_b":     round(sb, 4),
            "is_correct":  is_correct,
        })

    if (start // BATCH) % 10 == 0:
        done = min(start + BATCH, len(examples))
        acc  = sum(r["is_correct"] for r in results) / len(results) * 100
        print(f"  [{done}/{len(examples)}]  running accuracy: {acc:.1f}%")

df_results = pd.DataFrame(results)
df_results.to_csv(out_path, index=False)

correct = df_results["is_correct"].sum()
total   = len(df_results)
acc     = correct / total * 100
print(f"\nReward model on NYCC: {correct}/{total} = {acc:.1f}%")
print(f"Results saved to {out_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

row = {
    "model":        "qwen4b-humor-reward",
    "timestamp":    timestamp,
    "n_examples":   total,
    "overall_acc":  round(acc, 1),
    "unknown_count": 0,
    "dataset":      "nycc-reward",
}
summary_df = pd.DataFrame([row])
if SUMMARY_PATH.exists():
    summary_df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
else:
    summary_df.to_csv(SUMMARY_PATH, index=False)
print(f"Summary updated.")
