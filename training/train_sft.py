# ═══════════════════════════════════════════════════════════════════════════════
# SFT Fine-tuning: Gemma 4 E2B on Discord conversations
# ═══════════════════════════════════════════════════════════════════════════════
#
# GEMMA 4 GOTCHAS (from unsloth guide):
#
# 1. Loss of 13-15 is NORMAL for E2B/E4B — multimodal model quirk.
#    Don't stop training. 26B/31B have lower loss (1-3).
#
# 2. use_cache + gradient_checkpointing bug — when use_cache=False,
#    KV-shared layers (20 layers in E2B) produce garbage outputs and
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
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section 1: Model Loading ──────────────────────────────────────────────────
import torch
from unsloth import FastLanguageModel

MODEL_ID    = "google/gemma-4-E2B-it"
MAX_SEQ_LEN = 2048  # max tokens per training example — Discord convos are short
LORA_RANK   = 16    # expressiveness of LoRA adapters — higher = more capacity, more VRAM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,  # QLoRA: loads weights in 4-bit to save VRAM
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,       # scaling factor — keeping equal to rank is standard
    lora_dropout=0,             # 0 is recommended by unsloth for speed
    bias="none",
    use_gradient_checkpointing="unsloth",  # saves VRAM by recomputing activations
    random_state=42,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention — learns relationships between concepts
        "gate_proj", "up_proj", "down_proj",      # feedforward — stores pattern recognition
    ],
)

# ── Section 2: Dataset Formatting ─────────────────────────────────────────────
from datasets import load_dataset
from pathlib import Path
from unsloth.chat_templates import get_chat_template

DATA_PATH = Path(__file__).parent.parent / "datasets" / "discord" / "sft.jsonl"

# use unsloth's chat template — handles Gemma 4 quirks correctly
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
    # skip if first message isn't user or fewer than 2 turns after merging
    if len(messages) < 2 or messages[0]["role"] != "user":
        return {"text": ""}
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,               # return string, not token ids — SFTTrainer tokenizes later
        add_generation_prompt=False,  # no trailing prompt — we're training, not generating
        enable_thinking=False,        # disable thinking for E2B — use gemma-4-thinking for 26B/31B
    )
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=["messages"])
dataset = dataset.filter(lambda x: len(x["text"]) > 0)
print(f"Dataset size after filtering: {len(dataset)}")

# ── Section 3: Trainer Config ─────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

OUTPUT_DIR = str(Path(__file__).parent.parent / "outputs" / "gemma4-e2b-discord")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",          # column SFTTrainer reads from
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,      # effective batch size = 2 × 4 = 8
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=50,                    # ramp up LR slowly for first 50 steps
        bf16=torch.cuda.is_bf16_supported(),  # bf16 if GPU supports it (4090 does), else fp16
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=20,                   # print loss every 20 steps
        save_steps=500,                     # checkpoint every 500 steps
        save_total_limit=2,                 # keep only the 2 most recent checkpoints
        weight_decay=0.01,
        seed=42,
    ),
)

# ── Section 4: Training ───────────────────────────────────────────────────────
trainer_stats = trainer.train()
print(f"Training done. Final loss: {trainer_stats.training_loss:.4f}")

# ── Section 5: GGUF Export ────────────────────────────────────────────────────
GGUF_DIR = str(Path(__file__).parent.parent / "outputs" / "gemma4-e2b-discord-export")
# note: unsloth appends _gguf automatically → final dir is gemma4-e2b-discord-export_gguf

# merges LoRA into base model and exports as GGUF in one step
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")

print(f"GGUF exported to {GGUF_DIR}")
print("To load into Ollama, run:")
print(f'  ollama create gemma4-e2b-discord -f {GGUF_DIR}/Modelfile')
