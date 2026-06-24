"""
Fine-tune Qwen3.5-4B (instruct) with LoRA + regression head on Humicroedit subtask-1.
Trains jointly: LoRA adapters reshape Qwen representations, linear head maps to funniness score.
Same setup as our activation probe but with backbone unfrozen (PAI approach).
After training, evaluates on subtask-2 pairwise by comparing scores between headline pairs.

Run: python eval/train_lora_regression.py
"""
import re
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

DATA_DIR     = Path(__file__).parent.parent / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset"
S1_TRAIN     = DATA_DIR / "subtask-1" / "train.csv"
S2_TEST      = DATA_DIR / "subtask-2" / "test.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "probe"
CACHE_DIR    = Path(__file__).parent.parent / "results" / "probe" / "cache"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"
MODEL_SAVE   = CACHE_DIR / "lora_regression"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)


def apply_edit(original: str, edit: str) -> str:
    return re.sub(r"<[^/]+/>", edit, original).strip()


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading subtask-1 training data...")
s1        = pd.read_csv(S1_TRAIN)
headlines = [apply_edit(r.original, r.edit) for r in s1.itertuples()]
scores    = s1["meanGrade"].tolist()
print(f"  {len(headlines)} examples, meanGrade range {min(scores):.2f}–{max(scores):.2f}")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"   # consistent with probe: last token = assistant start token


def make_prompt(text: str) -> str:
    messages = [{"role": "user", "content": f"Consider the amount of funniness in the following: {text}"}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HeadlineDataset(Dataset):
    def __init__(self, headlines, scores):
        self.prompts = [make_prompt(h) for h in headlines]
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

dataset = HeadlineDataset(headlines, scores)
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

        if (step + 1) % 200 == 0:
            print(f"  Epoch {epoch+1} step {step+1}/{len(loader)}  loss={total_loss/(step+1):.4f}")

    print(f"Epoch {epoch+1} complete. Avg loss: {total_loss/len(loader):.4f}")

base.save_pretrained(MODEL_SAVE)
torch.save(head.state_dict(), MODEL_SAVE / "head.pt")
print(f"\nSaved LoRA adapters + head to {MODEL_SAVE}")


# ── Eval on subtask-2 pairwise ────────────────────────────────────────────────

print("\n── Eval: Humicroedit subtask-2 pairwise ──")
base.eval()
head.eval()

s2 = pd.read_csv(S2_TEST)
s2 = s2[s2["label"] != 0].reset_index(drop=True)
print(f"  {len(s2)} non-tie pairs")

headlines_a = [apply_edit(r.original1, r.edit1) for r in s2.itertuples()]
headlines_b = [apply_edit(r.original2, r.edit2) for r in s2.itertuples()]
all_prompts = [make_prompt(h) for h in headlines_a + headlines_b]


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
n          = len(s2)
scores_a   = all_scores[:n]
scores_b   = all_scores[n:]
predicted  = np.where(scores_a > scores_b, 1, 2)
acc        = (predicted == s2["label"].values).mean() * 100

print(f"\nSubtask-2 pairwise accuracy: {acc:.1f}%")
print(f"Baseline zero-shot: 56.3% | Frozen probe: 61.9% | This: {acc:.1f}%")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
row = {
    "model": "qwen4b-lora-regression", "timestamp": timestamp,
    "n_examples": len(s2), "overall_acc": round(acc, 1),
    "unknown_count": 0, "dataset": f"humicro-s2-pairwise-lora-layer{TARGET_LAYER}",
}
pd.DataFrame([row]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
print("Summary updated.")
