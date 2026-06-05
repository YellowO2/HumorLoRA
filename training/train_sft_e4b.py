# ═══════════════════════════════════════════════════════════════════════════════
# SFT Fine-tuning: Gemma 4 E4B on Discord conversations
# ═══════════════════════════════════════════════════════════════════════════════
#
# GEMMA 4 GOTCHAS (from unsloth guide):
#
# 1. Loss of 13-15 is NORMAL for E2B/E4B — multimodal model quirk.
#    Don't stop training. 26B/31B have lower loss (1-3).
#
# 2. use_cache + gradient_checkpointing bug — when use_cache=False,
#    KV-shared layers (20 layers in E4B) produce garbage outputs and
#    training diverges. Fixed by unsloth's gradient checkpointing:
#    use_gradient_checkpointing="unsloth" (NOT the HF default).
#
# 3. Chat template — use unsloth's get_chat_template("gemma-4") not
#    the raw HF tokenizer. Set enable_thinking=False for E2B/E4B.
#    Use "gemma-4-thinking" only for 26B/31B.
#
# 4. Gradient accumulation — standard HF trainer inflates loss when
#    using grad accum. Fixed in unsloth automatically.
#
# 5. GGUF/Ollama export is NOT used — KV-shared layer LoRA merge
#    corrupts weights. Use FastLanguageModel for inference (see eval/).
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section 1: Model Loading ──────────────────────────────────────────────────
import torch
from unsloth import FastLanguageModel

MODEL_ID    = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
MAX_SEQ_LEN = 2048
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

# ── Section 2: Dataset Formatting ─────────────────────────────────────────────
from datasets import load_dataset
from pathlib import Path
from unsloth.chat_templates import get_chat_template

DATA_PATH = Path(__file__).parent.parent / "datasets" / "discord" / "sft.jsonl"

tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

def merge_consecutive_roles(messages: list[dict]) -> list[dict]:
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append({"role": msg["role"], "content": msg["content"]})
    return merged


def format_example(example):
    messages = merge_consecutive_roles(example["messages"])
    if len(messages) < 2 or messages[0]["role"] != "user":
        return {"text": ""}
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=["messages"])
dataset = dataset.filter(lambda x: len(x["text"]) > 0)
print(f"Dataset size after filtering: {len(dataset)}")

# ── Section 3: Trainer Config ─────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

OUTPUT_DIR = str(Path(__file__).parent.parent / "outputs" / "gemma4-e4b-discord")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=1,      # reduced from E2B's 2 — E4B uses more VRAM
        gradient_accumulation_steps=8,      # effective batch size = 1 × 8 = 8 (same as E2B)
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=50,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        weight_decay=0.01,
        seed=42,
    ),
)

# ── Section 4: Training ───────────────────────────────────────────────────────
trainer_stats = trainer.train()
print(f"Training done. Final loss: {trainer_stats.training_loss:.4f}")
print(f"Checkpoint saved to {OUTPUT_DIR}")
print("To run inference, use eval/interact.py LocalModel with this checkpoint.")
