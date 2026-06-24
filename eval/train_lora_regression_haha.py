"""
Fine-tune Qwen3.5-4B (instruct) with LoRA + regression head on HaHackathon rating data.
Same architecture as train_lora_regression.py (Humicroedit version).
Training: humor_rating 0-5 from rating.csv (1000 examples), MSE loss.
Eval: pairwise.csv (2000 pairs) — compare predicted scores between joke pairs.

Run: python eval/train_lora_regression_haha.py
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/Qwen3.5-4B"
TARGET_LAYER = 16
LORA_R       = 16
LORA_ALPHA   = 32
LR           = 2e-4
EPOCHS       = 3
BATCH_SIZE   = 8
GRAD_ACCUM   = 2    # effective batch = 16
MAX_LENGTH   = 128
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent.parent / "datasets" / "hahackathon"
RATING_CSV   = DATA_DIR / "rating.csv"
PAIRWISE_CSV = DATA_DIR / "pairwise.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "probe"
CACHE_DIR    = Path(__file__).parent.parent / "results" / "probe" / "cache"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"
MODEL_SAVE   = CACHE_DIR / "lora_regression_haha"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading HaHa rating data...")
rating = pd.read_csv(RATING_CSV)
jokes  = rating["text"].tolist()
scores = rating["humor_rating"].tolist()
print(f"  {len(jokes)} examples, humor_rating range {min(scores):.2f}–{max(scores):.2f}")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"


def make_prompt(text: str) -> str:
    messages = [{"role": "user", "content": f"Consider the amount of funniness in the following: {text}"}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


# ── Dataset ───────────────────────────────────────────────────────────────────

class JokeDataset(Dataset):
    def __init__(self, jokes, scores):
        self.prompts = [make_prompt(j) for j in jokes]
        self.scores  = scores

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = tok(
            self.prompts[idx], truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.scores[idx], dtype=torch.float32),
        }


# ── Load model ────────────────────────────────────────────────────────────────

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
base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
base = get_peft_model(base, lora_cfg)
base.print_trainable_parameters()

hidden_size = base.config.hidden_size
head_device = next(base.parameters()).device
head        = nn.Linear(hidden_size, 1).to(head_device)

print(f"Model ready. Hidden size: {hidden_size}, head device: {head_device}")


# ── Training ──────────────────────────────────────────────────────────────────

dataset = JokeDataset(jokes, scores)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

optimizer   = torch.optim.AdamW(
    list(base.parameters()) + list(head.parameters()),
    lr=LR, weight_decay=0.01,
)
total_steps = (len(loader) // GRAD_ACCUM) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, total_steps // 10),
    num_training_steps=total_steps,
)

print(f"\nTraining {EPOCHS} epochs on {len(dataset)} examples...")
for epoch in range(EPOCHS):
    base.train()
    head.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(head_device)
        attention_mask = batch["attention_mask"].to(head_device)
        labels         = batch["label"].to(head_device)

        out  = base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h    = out.hidden_states[TARGET_LAYER][:, -1, :].to(head_device)
        pred = head(h.float()).squeeze(-1)
        loss = nn.MSELoss()(pred, labels) / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(
                list(base.parameters()) + list(head.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} step {step+1}/{len(loader)}  loss={total_loss/(step+1):.4f}")

    print(f"Epoch {epoch+1} complete. Avg loss: {total_loss/len(loader):.4f}")

base.save_pretrained(MODEL_SAVE)
torch.save(head.state_dict(), MODEL_SAVE / "head.pt")
print(f"\nSaved LoRA adapters + head to {MODEL_SAVE}")


# ── Eval on HaHa pairwise ─────────────────────────────────────────────────────

print("\n── Eval: HaHa pairwise ──")
base.eval()
head.eval()

pairwise = pd.read_csv(PAIRWISE_CSV)
print(f"  {len(pairwise)} pairs")

# text_a, text_b are included directly in pairwise.csv
jokes_a  = pairwise["text_a"].tolist()
jokes_b  = pairwise["text_b"].tolist()
expected = pairwise["expected"].tolist()  # "A" or "B"

all_prompts = [make_prompt(j) for j in jokes_a + jokes_b]


@torch.inference_mode()
def score_all(prompts):
    all_scores = []
    bs = BATCH_SIZE * 2
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        enc   = tok(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        enc   = {k: v.to(head_device) for k, v in enc.items()}
        out   = base(**enc, output_hidden_states=True)
        h     = out.hidden_states[TARGET_LAYER][:, -1, :].to(head_device)
        s     = head(h.float()).squeeze(-1)
        all_scores.extend(s.cpu().tolist())
        if i % (bs * 10) == 0:
            print(f"  {min(i + bs, len(prompts))}/{len(prompts)}")
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
