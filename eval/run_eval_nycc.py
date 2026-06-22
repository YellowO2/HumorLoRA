import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from interact import LocalModel, ask, unload

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = [
    # ── Ollama baselines (string = model name in Ollama) ──────────────────────
    # "gemma4:e2b", "gemma4:e4b", "qwen3.5:9b",
    # ── HF baselines (tuple = name, hf_model_id, chat_template) ──────────────
    # ("qwen3.5:4b", "unsloth/Qwen3.5-4B", "qwen-3"),
    # ── Fine-tuned LoRA checkpoints (tuple = name, local_path, chat_template) ─
    # ("gemma4-e2b-discord", str(OUTPUTS_DIR / "gemma4-e2b-discord" / "checkpoint-4011"), "gemma-4"),
    # ("gemma4-e4b-discord", str(OUTPUTS_DIR / "gemma4-e4b-discord" / "checkpoint-4011"), "gemma-4"),
    # ("qwen9b-discord",     str(OUTPUTS_DIR / "qwen9b-discord"     / "checkpoint-4011"), "qwen-3"),
    # ("qwen9b-degpt-dpo",   str(OUTPUTS_DIR / "qwen9b-degpt-dpo"   / "checkpoint-5592"), "qwen-3"),
    # ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct",   "llama-3.1"),
    # ("hermes-3-8b",         "NousResearch/Hermes-3-Llama-3.1-8B", "chatml"),
    # ("discord-hermes-3-8b", "mookiezii/Discord-Hermes-3-8B",      "chatml"),
    # ("qwen4b-degpt-dpo", str(OUTPUTS_DIR / "qwen4b-degpt-dpo" / "checkpoint-625"), "qwen-3"),
    ("qwen3.5:4b", "unsloth/Qwen3.5-4B", "qwen-3"),
]
N_EXAMPLES     = 2000  # upper bound of example range
EXAMPLE_OFFSET = 0     # skip first N examples (set to 0 to run from start)
DATASET        = "nycc"  # label written to summary.csv — change when switching datasets

# Each entry: (label_suffix, think_flag, instruction)
# Runs execute in order; model is reloaded between runs.
RUNS = [
    # crowd-framing prompt — testing if asking model to simulate crowd shifts accuracy
    ("-crowd", False, "Return only <answer>A</answer> or <answer>B</answer>."),
    # previous runs (kept for reference):
    # ("-no-gut", False, "Return <answer>A</answer> or <answer>B</answer>.")         # 54.5% n=2616
    # ("-thinking", True, "Briefly explain why each caption is funny or not, then return your final choice as <answer>A</answer> or <answer>B</answer>.")  # 53.2% n=2000
]
# ──────────────────────────────────────────────────────────────────────────────

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "newyorker"
RESULTS_DIR  = Path(__file__).parent.parent / "results"

# Set per-run in main() before each run_test call
_run_instruction: str = ""
_run_think: bool = False


def build_prompt(row: dict) -> str:
    return (
        "You are voting for which caption is funnier in a funny caption contest.\n\n"
        f"The cartoon shows: {row['image_description']}\n"
        f"Uncanny detail: {row['image_uncanny_description']}\n\n"
        "Two captions have been submitted:\n"
        f"A: {row['caption_a']}\n"
        f"B: {row['caption_b']}\n\n"
        f"Which caption do you think a crowd would find funnier if shared online? {_run_instruction}"
    )


def parse_response(content: str) -> str:
    import re
    m = re.search(r"<answer>([AB])</answer>", content, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    after_think = re.split(r"</think>", content, flags=re.IGNORECASE)[-1]
    text = after_think.strip() if after_think.strip() else content.strip()
    last_word = text.split()[-1] if text.split() else ""
    if re.fullmatch(r"[Aa]\.?", last_word):
        return "A"
    if re.fullmatch(r"[Bb]\.?", last_word):
        return "B"
    return "UNKNOWN"


def load_examples() -> pd.DataFrame:
    dfs = []
    for fold in range(5):
        df = pd.read_csv(DATASETS_DIR / f"fold{fold}_validation.csv")
        df["fold"] = fold
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    if N_EXAMPLES:
        combined = combined.iloc[EXAMPLE_OFFSET:N_EXAMPLES]
    return combined


SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"

def append_summary(model: str, results: list[dict], timestamp: str) -> None:
    summary_path = SUMMARY_PATH
    df = pd.DataFrame(results)
    row = {
        "model": model,
        "timestamp": timestamp,
        "n_examples": len(df),
        "overall_acc": round(df["is_correct"].mean() * 100, 1),
        "unknown_count": int((df["prediction"] == "UNKNOWN").sum()),
        "dataset": DATASET,
    }
    summary_df = pd.DataFrame([row])
    if summary_path.exists():
        summary_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)
    print(f"Summary updated at {summary_path}")


def run_test(model, examples: pd.DataFrame, label_suffix: str) -> None:
    is_local = isinstance(model, LocalModel)
    model_name = (model.name if is_local else model) + label_suffix

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{model_name.replace(':', '_')}_{timestamp}.csv"

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  {len(examples)} examples")
    print(f"{'='*60}")

    results = []
    for i, row in enumerate(examples.itertuples()):
        prompt = build_prompt(row._asdict())
        max_tokens = 2048 if _run_think else 512
        out = model.ask(prompt, think=_run_think, history=None, max_new_tokens=max_tokens) if is_local else ask(prompt, model=model, think=_run_think)
        prediction = parse_response(out["content"])
        expected = str(row.expected).strip().upper()
        is_correct = prediction == expected

        import re as _re
        think_match = _re.search(r"<think>(.*?)</think>", out['content'], _re.DOTALL | _re.IGNORECASE)
        think_text = think_match.group(1).strip() if think_match else ""
        after_think = _re.split(r"</think>", out['content'], flags=_re.IGNORECASE)[-1].strip()
        print(f"  [{i+1}] fold={row.fold} pred={prediction} expected={expected} {'✓' if is_correct else '✗'}")
        print(f"       think({len(think_text)} chars): {think_text[:200].replace(chr(10), ' ')!r}")
        print(f"       answer: {after_think!r}")

        results.append({
            "fold": row.fold,
            "sample_id": row.sample_id,
            "expected": expected,
            "prediction": prediction,
            "is_correct": is_correct,
            "raw_response": out["content"],
            "thinking": out["thinking"],
        })

    pd.DataFrame(results).to_csv(out_path, index=False)
    correct = sum(r["is_correct"] for r in results)
    print(f"\n{model_name} overall: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    print(f"Results saved to {out_path}")
    append_summary(model_name, results, timestamp)
    model.unload() if is_local else unload(model)
    print(f"Unloaded {model_name} from VRAM")


def main():
    global _run_instruction, _run_think
    examples = load_examples()
    print(f"Loaded {len(examples)} examples across {examples['fold'].nunique()} folds")

    for label_suffix, think, instruction in RUNS:
        _run_instruction = instruction
        _run_think = think
        print(f"\n{'#'*60}")
        print(f"Run: {label_suffix}  |  think={think}")
        print(f"Instruction: {instruction}")
        print(f"{'#'*60}")
        for model_spec in MODELS:
            if isinstance(model_spec, tuple):
                name, checkpoint, chat_template = model_spec
                model = LocalModel(name, checkpoint, chat_template=chat_template, enable_thinking=False)
            else:
                model = model_spec
            run_test(model, examples, label_suffix)


if __name__ == "__main__":
    main()
