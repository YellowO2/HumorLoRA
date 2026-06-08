import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUTS = Path(__file__).parent / "outputs"

CHECKPOINTS = {
    "e2b": str(OUTPUTS / "gemma4-e2b-discord" / "checkpoint-4011"),
    "e4b": str(OUTPUTS / "gemma4-e4b-discord" / "checkpoint-4011"),
    "dpo":  str(OUTPUTS / "qwen9b-degpt-dpo"   / "checkpoint-5592"),
    "dpo4b": str(OUTPUTS / "qwen4b-degpt-dpo"  / "checkpoint-625"),
}

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "dpo4b"
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else CHECKPOINTS.get(model_key)

    if not checkpoint:
        print(f"Unknown model '{model_key}'. Pass a checkpoint path as second arg.")
        sys.exit(1)

    print(f"Loading {model_key} from {checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Ready. Type your message and press Enter. Ctrl+C or 'quit' to exit.\n")

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Bye.")
            break

        history.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"Model: {reply}\n")


if __name__ == "__main__":
    main()
