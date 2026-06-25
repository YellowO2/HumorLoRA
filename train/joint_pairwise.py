"""
Joint pairwise LoRA training across multiple humor datasets.

Architecture: same as eval/train_lora_pairwise_haha.py
  - Qwen3.5-4B, 4-bit, LoRA r=16 on attn projections
  - Layer-16 hidden state → linear scalar head
  - In-batch pairwise BCE loss

Datasets (all unified to prompt_text + score format):
  - hahackathon   (humor_rating 0–5)
  - humicroedit   (meanGrade 0–3)
  - reddit_jokes  (upvote_ratio 0.55–1.0)
  - haha_spanish  (funniness_average 0–5)
  - humor_arena   (pairwise A/B winners → score 1.0 / 0.0)

Pairwise pairs are drawn WITHIN each source only — no cross-dataset comparisons.
This avoids the scale normalization problem entirely.

Run: python train/joint_pairwise.py
  or: python train/joint_pairwise.py --datasets hahackathon humicroedit reddit_jokes
"""
import argparse
import gc
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
EPOCHS           = 3
BATCH_SIZE       = 16   # larger batch → more pairs per step across datasets
GRAD_ACCUM       = 2
MAX_LENGTH       = 192  # longer than single-joke to handle humicroedit two-line format
SEED             = 42
CAP_PER_DATASET  = 10000  # stratified cap per dataset; datasets smaller than this use all rows
# ─────────────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
LORA_DATA_DIR = ROOT / "datasets" / "lora_train_data"
CACHE_DIR     = ROOT / "results" / "probe" / "cache"
MODEL_SAVE = CACHE_DIR / "lora_joint_pairwise"
SUMMARY_PATH = ROOT / "results" / "summary.csv"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+",
                    default=["hahackathon", "humicroedit", "reddit_jokes", "haha_spanish", "nycc"],
                    help="Datasets to include in joint training")
args = parser.parse_args()


# ── Load and unify datasets ───────────────────────────────────────────────────

def load_dataset(name: str) -> pd.DataFrame:
    p = LORA_DATA_DIR / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Run prepare/{name}.py first: {p}")
    df = pd.read_csv(p)[["prompt_text", "score", "source"]]
    if len(df) <= CAP_PER_DATASET:
        return df
    # Stratified sample: bin scores into 10 quantiles, sample proportionally
    df["_bin"] = pd.qcut(df["score"], q=10, labels=False, duplicates="drop")
    df = (
        df.groupby("_bin", group_keys=False)
        .apply(lambda g: g.sample(frac=CAP_PER_DATASET / len(df), random_state=SEED))
    )
    df = df.drop(columns=["_bin"]).reset_index(drop=True)
    return df


print(f"Loading datasets (cap={CAP_PER_DATASET} per dataset): {args.datasets}")
dfs = []
for name in args.datasets:
    try:
        df = load_dataset(name)
        dfs.append(df)
        print(f"  {name}: {len(df)} rows, score {df['score'].min():.2f}–{df['score'].max():.2f}")
    except FileNotFoundError as e:
        print(f"  SKIP {name}: {e}")

all_data = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(all_data)} examples across {all_data['source'].nunique()} datasets")
print(all_data["source"].value_counts().to_string())


# ── Tokenizer ─────────────────────────────────────────────────────────────────

print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"


def make_chat_prompt(prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


# ── Dataset ───────────────────────────────────────────────────────────────────

class JokeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.prompts = [make_chat_prompt(t) for t in df["prompt_text"].tolist()]
        self.scores  = df["score"].tolist()
        self.sources = df["source"].tolist()

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
            "score":          torch.tensor(self.scores[idx], dtype=torch.float32),
            "source":         self.sources[idx],
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "score":          torch.stack([b["score"] for b in batch]),
        "source":         [b["source"] for b in batch],
    }


# ── In-batch pairwise loss (within-source only) ───────────────────────────────

def pairwise_bce_loss(pred_scores, true_scores, sources):
    """
    Same in-batch BCE as single-dataset version, but pairs are only formed
    between items from the same source dataset. This avoids cross-scale issues
    (e.g., humicroedit 0–3 vs hahackathon 0–5).
    """
    N = pred_scores.size(0)
    diff_pred = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)   # (N, N)
    diff_true = true_scores.unsqueeze(1) - true_scores.unsqueeze(0)   # (N, N)

    # Same-source mask
    same_source = torch.zeros(N, N, dtype=torch.bool, device=pred_scores.device)
    for i in range(N):
        for j in range(N):
            if sources[i] == sources[j]:
                same_source[i, j] = True

    upper = torch.triu(torch.ones(N, N, device=pred_scores.device), diagonal=1).bool()
    mask  = upper & same_source & (diff_true != 0)

    if mask.sum() == 0:
        return pred_scores.sum() * 0.0

    logits = diff_pred[mask]
    labels = (diff_true[mask] > 0).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


# ── Load model ────────────────────────────────────────────────────────────────

print("Loading Qwen in 4-bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto", local_files_only=True,
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

dataset = JokeDataset(all_data.sample(frac=1, random_state=SEED).reset_index(drop=True))
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=2, collate_fn=collate_fn)

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

print(f"\nTraining {EPOCHS} epochs on {len(dataset)} examples (batch={BATCH_SIZE})...")
for epoch in range(EPOCHS):
    base.train()
    head.train()
    total_loss = 0.0
    valid_steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(head_device)
        attention_mask = batch["attention_mask"].to(head_device)
        true_scores    = batch["score"].to(head_device)
        sources        = batch["source"]

        out  = base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h    = out.hidden_states[TARGET_LAYER][:, -1, :].to(head_device)
        pred = head(h.float()).squeeze(-1)

        loss = pairwise_bce_loss(pred, true_scores, sources) / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM
        valid_steps += 1

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(
                list(base.parameters()) + list(head.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 50 == 0:
            avg = total_loss / valid_steps if valid_steps else 0
            print(f"  Epoch {epoch+1} step {step+1}/{len(loader)}  loss={avg:.4f}")

    avg = total_loss / valid_steps if valid_steps else 0
    print(f"Epoch {epoch+1} complete. Avg loss: {avg:.4f}")

base.save_pretrained(MODEL_SAVE)
torch.save(head.state_dict(), MODEL_SAVE / "head.pt")
print(f"\nSaved to {MODEL_SAVE}")

del optimizer, scheduler, dataset, loader
gc.collect()
torch.cuda.empty_cache()


# ── Eval on HaHa pairwise (same benchmark as single-dataset runs) ─────────────

print("\n── Eval: HaHa pairwise held-out ──")
base.eval()
head.eval()

pairwise = pd.read_csv(ROOT / "datasets" / "hahackathon" / "pairwise.csv")
print(f"  {len(pairwise)} pairs")

jokes_a  = pairwise["text_a"].tolist()
jokes_b  = pairwise["text_b"].tolist()
expected = pairwise["expected"].tolist()

prompt_prefix = "Consider the amount of funniness in the following: "
all_prompts = [make_chat_prompt(prompt_prefix + j) for j in jokes_a + jokes_b]


@torch.inference_mode()
def score_all(prompts):
    all_scores = []
    bs = BATCH_SIZE
    for i in range(0, len(prompts), bs):
        batch_p = prompts[i:i + bs]
        enc     = tok(batch_p, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        enc     = {k: v.to(head_device) for k, v in enc.items()}
        out     = base(**enc, output_hidden_states=True)
        h       = out.hidden_states[TARGET_LAYER][:, -1, :].to(head_device)
        s       = head(h.float()).squeeze(-1)
        all_scores.extend(s.cpu().tolist())
        if i % (bs * 20) == 0:
            print(f"  {min(i + bs, len(prompts))}/{len(prompts)}")
    return np.array(all_scores)


scores_all = score_all(all_prompts)
n          = len(pairwise)
scores_a   = scores_all[:n]
scores_b   = scores_all[n:]
predicted  = np.where(scores_a > scores_b, "A", "B")
acc        = (predicted == np.array(expected)).mean() * 100

print(f"\nHaHa pairwise accuracy: {acc:.1f}%  (n={n})")
print(f"Single-dataset baselines: regression=68.3%, pairwise-haha=67.7%")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
row = {
    "model": "qwen4b-lora-joint-pairwise", "timestamp": timestamp,
    "n_examples": n, "overall_acc": round(acc, 1),
    "unknown_count": 0, "dataset": f"joint-pairwise-layer{TARGET_LAYER}",
}
pd.DataFrame([row]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
print("Summary updated.")
