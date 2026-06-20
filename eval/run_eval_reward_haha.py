# ═══════════════════════════════════════════════════════════════════════════════
# Eval: humor reward model on held-out HaHa jokes (in-domain test)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tests whether the reward model learned funniness on complete self-contained
# jokes it was not trained on (15% HaHa held-out split).
#
# Run: python eval/run_eval_reward_haha.py
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

CHECKPOINT   = Path(__file__).parent.parent / "outputs" / "qwen4b-humor-reward" / "best"
BASE_MODEL   = "unsloth/Qwen3.5-4B"
TEST_PATH    = Path(__file__).parent.parent / "datasets" / "reward" / "test_haha.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "reward"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"
MAX_LENGTH   = 256


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
    if logits.dim() > 1 and logits.shape[-1] > 1:
        logits = logits[:, 0]
    else:
        logits = logits.squeeze(-1)
    return logits.float().cpu().tolist()


# ── Load test set ─────────────────────────────────────────────────────────────

examples = pd.read_csv(TEST_PATH)
print(f"\nLoaded {len(examples)} held-out HaHa pairs")


# ── Eval ──────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = RESULTS_DIR / f"reward_haha_{timestamp}.csv"

results = []
BATCH   = 16

for start in range(0, len(examples), BATCH):
    batch    = examples.iloc[start:start + BATCH]
    scores_a = score(batch["chosen"].tolist())
    scores_b = score(batch["rejected"].tolist())

    for i, row in enumerate(batch.itertuples()):
        sa, sb     = scores_a[i], scores_b[i]
        prediction = "chosen" if sa > sb else "rejected"
        is_correct = prediction == "chosen"   # chosen is always the funnier one
        results.append({
            "chosen":      row.chosen,
            "rejected":    row.rejected,
            "score_chosen":   round(sa, 4),
            "score_rejected": round(sb, 4),
            "is_correct":  is_correct,
        })

    if (start // BATCH) % 5 == 0:
        done = min(start + BATCH, len(examples))
        acc  = sum(r["is_correct"] for r in results) / len(results) * 100
        print(f"  [{done}/{len(examples)}]  running accuracy: {acc:.1f}%")

df_results = pd.DataFrame(results)
df_results.to_csv(out_path, index=False)

correct = df_results["is_correct"].sum()
total   = len(df_results)
acc     = correct / total * 100
print(f"\nReward model on held-out HaHa: {correct}/{total} = {acc:.1f}%")
print(f"Results saved to {out_path}")

row = {
    "model":        "qwen4b-humor-reward",
    "timestamp":    timestamp,
    "n_examples":   total,
    "overall_acc":  round(acc, 1),
    "unknown_count": 0,
    "dataset":      "haha-reward-test",
}
summary_df = pd.DataFrame([row])
if SUMMARY_PATH.exists():
    summary_df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
else:
    summary_df.to_csv(SUMMARY_PATH, index=False)
print("Summary updated.")
