"""
Eval-only: load saved HaHa LoRA regression model and score pairwise.csv.
Assumes train_lora_regression_haha.py has already run and saved to CACHE_DIR/lora_regression_haha/.

Run: python eval/eval_lora_regression_haha.py
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/Qwen3.5-4B"
TARGET_LAYER = 16
BATCH_SIZE   = 8
MAX_LENGTH   = 128
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent.parent / "datasets" / "hahackathon"
PAIRWISE_CSV = DATA_DIR / "pairwise.csv"
CACHE_DIR    = Path(__file__).parent.parent / "results" / "probe" / "cache"
MODEL_SAVE   = CACHE_DIR / "lora_regression_haha"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"


# ── Load model + head ─────────────────────────────────────────────────────────

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

print("Loading Qwen in 4-bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto",
)
base = PeftModel.from_pretrained(base, MODEL_SAVE)
base.eval()

head_device = next(base.parameters()).device
hidden_size = base.config.hidden_size
head = nn.Linear(hidden_size, 1).to(head_device)
head.load_state_dict(torch.load(MODEL_SAVE / "head.pt", map_location=head_device))
head.eval()

print(f"Model ready on {head_device}")


# ── Eval ──────────────────────────────────────────────────────────────────────

def make_prompt(text: str) -> str:
    messages = [{"role": "user", "content": f"Consider the amount of funniness in the following: {text}"}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


print("\n── Eval: HaHa pairwise ──")
pairwise = pd.read_csv(PAIRWISE_CSV)
print(f"  {len(pairwise)} pairs")

jokes_a  = pairwise["text_a"].tolist()
jokes_b  = pairwise["text_b"].tolist()
expected = pairwise["expected"].tolist()

all_prompts = [make_prompt(j) for j in jokes_a + jokes_b]


@torch.inference_mode()
def score_all(prompts):
    all_scores = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        enc   = tok(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        enc   = {k: v.to(head_device) for k, v in enc.items()}
        out   = base(**enc, output_hidden_states=True)
        h     = out.hidden_states[TARGET_LAYER][:, -1, :].to(head_device)
        s     = head(h.float()).squeeze(-1)
        all_scores.extend(s.cpu().tolist())
        if i % (BATCH_SIZE * 20) == 0:
            print(f"  {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}")
    return np.array(all_scores)


all_scores = score_all(all_prompts)
n          = len(pairwise)
scores_a   = all_scores[:n]
scores_b   = all_scores[n:]
predicted  = np.where(scores_a > scores_b, "A", "B")
acc        = (predicted == np.array(expected)).mean() * 100

print(f"\nHaHa pairwise accuracy: {acc:.1f}%  (n={n})")
print(f"Cross-domain frozen probe baseline: 55.5%  |  This: {acc:.1f}%")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
row = {
    "model": "qwen4b-lora-regression-haha", "timestamp": timestamp,
    "n_examples": n, "overall_acc": round(acc, 1),
    "unknown_count": 0, "dataset": f"haha-pairwise-lora-layer{TARGET_LAYER}",
}
pd.DataFrame([row]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
print("Summary updated.")
