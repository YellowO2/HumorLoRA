from pathlib import Path
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

CHECKPOINT = str(Path(__file__).parent.parent / "outputs" / "gemma4-e2b-discord" / "checkpoint-4011")
GGUF_DIR   = str(Path(__file__).parent.parent / "outputs" / "gemma4-e2b-discord-export")
# unsloth appends _gguf → final dir: gemma4-e2b-discord-export_gguf

model, tokenizer = FastModel.from_pretrained(
    model_name=CHECKPOINT,
    max_seq_length=2048,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="bf16")
print(f"Exported to {GGUF_DIR}_gguf")
