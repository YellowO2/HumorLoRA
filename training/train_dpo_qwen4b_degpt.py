# ═══════════════════════════════════════════════════════════════════════════════
# DPO Fine-tuning: Qwen 3.5 4B on De-GPT-DPO (human vs AI preference)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Hypothesis: nudging the model to prefer human-style responses over AI-style
# responses will align it closer to human taste, which should transfer to
# downstream human preference tasks (e.g. NYCC humor judgment).
#
# Dataset: qingy2024/De-GPT-DPO (44.7k rows)
#   chosen   = real human-written response (short, casual, direct)
#   rejected = AI/GPT-style response (verbose, structured, formal)
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section 1: Model Loading ──────────────────────────────────────────────────
import os
import torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from unsloth import FastLanguageModel

MODEL_ID    = "unsloth/Qwen3.5-4B"
MAX_SEQ_LEN = 512
LORA_RANK   = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ── Section 2: Dataset ────────────────────────────────────────────────────────
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")

raw = load_dataset("qingy2024/De-GPT-DPO", split="train")


def to_pair(example):
    chosen_text   = example["chosen"][-1]["content"]
    rejected_text = example["rejected"][-1]["content"]
    return {
        "prompt":   [{"role": "user",      "content": example["prompt"]}],
        "chosen":   [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
    }


dataset = raw.map(to_pair, remove_columns=raw.column_names).select(range(5000))
print(f"DPO pairs: {len(dataset)}")

# ── Section 3: Trainer Config ─────────────────────────────────────────────────
from pathlib import Path

from trl import DPOConfig, DPOTrainer
from unsloth import PatchDPOTrainer
PatchDPOTrainer()

OUTPUT_DIR = str(Path(__file__).parent.parent / "outputs" / "qwen4b-degpt-dpo")

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-6,
        beta=0.1,
        warmup_steps=50,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=512,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        weight_decay=0.01,
        seed=42,
        remove_unused_columns=False,
    ),
)

# ── Section 4: Training ───────────────────────────────────────────────────────
RESUME = str(Path(__file__).parent.parent / "outputs" / "qwen4b-degpt-dpo" / "checkpoint-500")  # ~2500 total steps
stats = trainer.train(resume_from_checkpoint=RESUME if Path(RESUME).exists() else None)
print(f"Training done. Final loss: {stats.training_loss:.4f}")

# ── Section 5: GGUF Export ────────────────────────────────────────────────────
GGUF_DIR = str(Path(__file__).parent.parent / "outputs" / "qwen4b-degpt-dpo-export")
try:
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
    print(f"GGUF exported to {GGUF_DIR}_gguf")
    print(f"  ollama create qwen4b-degpt-dpo -f {GGUF_DIR}_gguf/Modelfile")
except Exception as e:
    print(f"GGUF export failed (expected, see README): {e}")
