"""
Crowd Score evaluation on HaHa pairwise benchmark.

Implements Goes et al. (2022) "Crowd Score" method:
  - 4 humor personality types (Martin et al. 2003 HSQ): affiliative, self-enhancing, aggressive, self-defeating
  - Each personality votes Funny/Boring for each joke (zero-shot, as per paper — few-shot overwrites personality)
  - Crowd score = sum of Funny votes (0–4 per joke)
  - Pairwise prediction: joke with higher crowd score wins
  - Ties broken randomly (50/50)

Skips auditing step for simplicity (adds 2x LLM calls with marginal benefit per paper).

Usage: python eval/eval_crowd_score.py
"""
import random
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "unsloth/Qwen3.5-4B"
MAX_NEW_TOKENS = 5
SEED = 42

ROOT         = Path(__file__).parent.parent
SUMMARY_PATH = ROOT / "results" / "summary.csv"

PERSONALITIES = ["affiliative", "self-enhancing", "aggressive", "self-defeating"]

random.seed(SEED)
torch.manual_seed(SEED)

print("Loading tokenizer and model...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()
dev = next(model.parameters()).device
print(f"Model loaded on {dev}")


def build_prompt(joke: str, personality: str) -> str:
    """Figure 3 from Goes et al. (2022) — zero-shot personality induction."""
    content = (
        f"Classify the following joke as Funny or Boring "
        f"as a person that enjoys {personality} humour.\n\n"
        f"Joke: {joke}\n"
        f"Classification:"
    )
    messages = [{"role": "user", "content": content}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@torch.inference_mode()
def is_funny(joke: str, personality: str) -> bool:
    """Returns True if this personality votes the joke as Funny."""
    prompt = build_prompt(joke, personality)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(dev)
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    generated = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if not hasattr(is_funny, "_debug_count"):
        is_funny._debug_count = 0
    if is_funny._debug_count < 8:
        print(f"  [debug] personality={personality!r} → {generated!r}")
        is_funny._debug_count += 1
    return generated.lower().startswith("funny")


def crowd_score(joke: str) -> int:
    """Sum of Funny votes across all 4 personalities (range 0–4)."""
    return sum(is_funny(joke, p) for p in PERSONALITIES)


# ── HaHa pairwise eval ────────────────────────────────────────────────────────

pairwise_path = ROOT / "datasets" / "hahackathon" / "pairwise.csv"
pw = pd.read_csv(pairwise_path)
print(f"\nEvaluating {len(pw)} pairs on HaHa pairwise benchmark...")
print(f"4 personalities × 2 jokes × {len(pw)} pairs = {4*2*len(pw)} LLM calls\n")

correct = 0
ties = 0

for i, row in pw.iterrows():
    joke_a = row["text_a"]
    joke_b = row["text_b"]

    score_a = crowd_score(joke_a)
    score_b = crowd_score(joke_b)

    if score_a > score_b:
        pred = "A"
    elif score_b > score_a:
        pred = "B"
    else:
        pred = random.choice(["A", "B"])
        ties += 1

    if pred == row["expected"]:
        correct += 1

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(pw)}  running acc: {correct/(i+1)*100:.1f}%  ties so far: {ties}")

n = len(pw)
acc = correct / n * 100
print(f"\nCrowd Score accuracy: {acc:.1f}%  (n={n}, ties={ties})")
print(f"Baseline zero-shot: ~55.4%")
print(f"Our joint LoRA:      65.2%")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
row = {
    "model": "qwen4b-crowd-score-4persona",
    "timestamp": timestamp,
    "n_examples": n,
    "overall_acc": round(acc, 1),
    "unknown_count": ties,
    "dataset": "crowd-score-goes2022",
}
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
print(f"Logged to {SUMMARY_PATH}")
