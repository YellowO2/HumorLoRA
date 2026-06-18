# ═══════════════════════════════════════════════════════════════════════════════
# Humor Reward Model: Qwen3.5-4B regression head
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture: Qwen3.5-4B (frozen) + LoRA + linear(1) regression head
# Input:  single joke text
# Output: scalar funniness score in [0, 1]
# Loss:   MSE against normalized human ratings
#
# Training data: HaHackathon + Jester (datasets/reward/train.csv)
# Test set:      NYCC pairwise (held out — evaluated separately)
#
# Run: python training/train_reward_model.py
# ═══════════════════════════════════════════════════════════════════════════════

import os
import torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID    = "unsloth/Qwen3.5-4B"
MAX_LENGTH  = 256
LORA_RANK   = 16
BATCH_SIZE  = 4
GRAD_ACCUM  = 4
EPOCHS      = 3
LR          = 2e-4
SEED        = 42

DATA_PATH   = Path(__file__).parent.parent / "datasets" / "reward" / "train.csv"
OUTPUT_DIR  = Path(__file__).parent.parent / "outputs" / "qwen4b-humor-reward"

# ── Section 1: Dataset ────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} examples  |  score mean={df['score'].mean():.3f}  std={df['score'].std():.3f}")

train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

# ── Section 2: Tokenizer ──────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    enc["labels"] = batch["score"]
    return enc

train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text", "score", "source"])
val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["text", "score", "source"])
train_ds.set_format("torch")
val_ds.set_format("torch")

# ── Section 3: Model + LoRA ───────────────────────────────────────────────────

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["score"],   # always train the regression head
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ── Section 4: Custom Trainer (MSE loss) ──────────────────────────────────────

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        logits = logits.squeeze(-1)
        mse  = float(np.mean((logits - labels) ** 2))
        rmse = float(np.sqrt(mse))
        # Spearman r between predicted scores and human ratings
        from scipy.stats import spearmanr
        r, _ = spearmanr(logits, labels)
        return {"rmse": rmse, "spearman_r": r}

# ── Section 5: Training ───────────────────────────────────────────────────────

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=50,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="spearman_r",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
)

trainer = RegressionTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

stats = trainer.train()
print(f"\nTraining done. Final loss: {stats.training_loss:.4f}")

# ── Section 6: Save ───────────────────────────────────────────────────────────

trainer.save_model(str(OUTPUT_DIR / "best"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "best"))
print(f"Model saved to {OUTPUT_DIR / 'best'}")
