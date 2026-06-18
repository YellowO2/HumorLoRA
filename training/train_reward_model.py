# ═══════════════════════════════════════════════════════════════════════════════
# Humor Reward Model: Qwen3.5-4B + LoRA, trained with TRL RewardTrainer
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture: Qwen3.5-4B + LoRA + linear(1) score head (num_labels forced to 1 by TRL)
# Training:     Bradley-Terry pairwise loss on HaHa + Jester joke pairs
# Inference:    single joke text → scalar funniness score (relative ordering)
# Test set:     NYCC pairwise (held out — different domain, evaluated separately)
#
# Run: python training/train_reward_model.py
# ═══════════════════════════════════════════════════════════════════════════════

import os
import torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import RewardTrainer, RewardConfig

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID   = "unsloth/Qwen3.5-4B"
MAX_LENGTH = 256
LORA_RANK  = 16
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS     = 3
LR         = 2e-4
SEED       = 42

DATA_PATH  = Path(__file__).parent.parent / "datasets" / "reward" / "train.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "qwen4b-humor-reward"

# ── Section 1: Dataset ────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} pairs  |  sources: {df['source'].value_counts().to_dict()}")

train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

# RewardTrainer expects 'chosen' and 'rejected' columns as plain strings
train_ds = Dataset.from_pandas(train_df[["chosen", "rejected"]].reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df[["chosen", "rejected"]].reset_index(drop=True))

# ── Section 2: LoRA config ────────────────────────────────────────────────────

peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["score"],   # score head must be trained alongside LoRA
)

# ── Section 3: RewardTrainer ──────────────────────────────────────────────────
# RewardTrainer automatically sets num_labels=1 when loading from model string.
# Pass quantization config via model_init_kwargs so it loads in 4-bit.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

trainer = RewardTrainer(
    model=MODEL_ID,
    args=RewardConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_steps=50,
        max_length=MAX_LENGTH,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        center_rewards_coefficient=0.01,
        seed=SEED,
        report_to="none",
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
        },
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
)

# ── Section 4: Train ──────────────────────────────────────────────────────────

stats = trainer.train()
print(f"\nTraining done. Final loss: {stats.training_loss:.4f}")

trainer.save_model(str(OUTPUT_DIR / "best"))
print(f"Model saved to {OUTPUT_DIR / 'best'}")
