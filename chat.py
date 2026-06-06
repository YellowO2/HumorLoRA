import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "eval"))
from interact import LocalModel

OUTPUTS = Path(__file__).parent / "outputs"

# (checkpoint_path, chat_template)
CHECKPOINTS = {
    "e2b": (str(OUTPUTS / "gemma4-e2b-discord" / "checkpoint-4011"), "gemma-4"),
    "e4b": (str(OUTPUTS / "gemma4-e4b-discord" / "checkpoint-4011"), "gemma-4"),
    "dpo": (str(OUTPUTS / "qwen9b-degpt-dpo"   / "checkpoint-5592"), "qwen-3"),
}

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "e2b"
    entry = CHECKPOINTS.get(model_key)
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else (entry[0] if entry else None)
    chat_template = entry[1] if entry else ("qwen-3" if "qwen" in model_key.lower() else "gemma-4")

    if not checkpoint:
        print(f"Unknown model '{model_key}'. Use: e2b, e4b, dpo, or pass a checkpoint path.")
        sys.exit(1)

    print(f"Loading {model_key} from {checkpoint} ...")
    model = LocalModel(model_key, checkpoint, chat_template=chat_template)
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

        result = model.ask(user_input, history=history)
        reply = result["content"]
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        print(f"Model: {reply}\n")


if __name__ == "__main__":
    main()
