# ═══════════════════════════════════════════════════════════════════════════════
# SFT Fine-tuning: Qwen 3.5 9B on Discord conversations
# ═══════════════════════════════════════════════════════════════════════════════
#
# Unlike Gemma 4 E2B/E4B, Qwen has standard architecture (no KV-shared layers),
# so LoRA merge and GGUF export work correctly → can be loaded into Ollama.
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section 1: Model Loading ──────────────────────────────────────────────────
import torch
from unsloth import FastLanguageModel

MODEL_ID    = "unsloth/Qwen3.5-9B"
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

tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")

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
        enable_thinking=False,  # disable <think> tokens — Qwen3.5 thinks by default
    )
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=["messages"])
dataset = dataset.filter(lambda x: len(x["text"]) > 0)
print(f"Dataset size after filtering: {len(dataset)}")

# ── Section 3: Trainer Config ─────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

OUTPUT_DIR = str(Path(__file__).parent.parent / "outputs" / "qwen9b-discord")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,      # effective batch size = 1 × 8 = 8
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

# ── Section 5: GGUF Export (Ollama-compatible) ────────────────────────────────
GGUF_DIR = str(Path(__file__).parent.parent / "outputs" / "qwen9b-discord-export")

model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
print(f"GGUF exported to {GGUF_DIR}_gguf")
print("To load into Ollama:")
print(f"  ollama create qwen9b-discord -f {GGUF_DIR}_gguf/Modelfile")
