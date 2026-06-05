import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "eval"))
from interact import LocalModel

CHECKPOINTS = {
    "e2b": str(Path(__file__).parent / "outputs" / "gemma4-e2b-discord" / "checkpoint-4011"),
    "e4b": str(Path(__file__).parent / "outputs" / "gemma4-e4b-discord" / "checkpoint-last"),
}

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "e2b"
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else CHECKPOINTS.get(model_key)

    if not checkpoint:
        print(f"Unknown model '{model_key}'. Use: e2b, e4b, or pass a checkpoint path directly.")
        sys.exit(1)

    print(f"Loading {model_key} from {checkpoint} ...")
    model = LocalModel(model_key, checkpoint)
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
