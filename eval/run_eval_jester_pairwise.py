import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from interact import LocalModel, ask, unload

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = [
    ("qwen3.5:4b",        "Qwen/Qwen2.5-3B-Instruct",                    "qwen2"),
    ("qwen4b-degpt-dpo",  "outputs/qwen4b-degpt-dpo/checkpoint-625",      "qwen2"),
]
DATASET = "jester-pairwise"

RUNS = [
    ("-no-gut", False, "Return only <answer>A</answer> or <answer>B</answer>."),
]
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH    = Path(__file__).parent.parent / "datasets" / "jester" / "pairwise.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "jester"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"

_run_instruction: str = ""
_run_think: bool = False


def build_prompt(row: dict) -> str:
    return (
        "You are judging which of two texts is funnier.\n\n"
        f"A: {row['text_a']}\n"
        f"B: {row['text_b']}\n\n"
        f"Which do you find funnier? {_run_instruction}"
    )


def parse_response(content: str) -> str:
    m = re.search(r"<answer>([AB])</answer>", content, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    after_think = re.split(r"</think>", content, flags=re.IGNORECASE)[-1].strip()
    text = after_think if after_think else content.strip()
    last = text.split()[-1] if text.split() else ""
    if re.fullmatch(r"[Aa]\.?", last):
        return "A"
    if re.fullmatch(r"[Bb]\.?", last):
        return "B"
    return "UNKNOWN"


def append_summary(model: str, results: list[dict], timestamp: str) -> None:
    df = pd.DataFrame(results)
    row = {
        "model": model, "timestamp": timestamp,
        "n_examples": len(df),
        "overall_acc": round(df["is_correct"].mean() * 100, 1),
        "unknown_count": int((df["prediction"] == "UNKNOWN").sum()),
        "dataset": DATASET,
    }
    summary_df = pd.DataFrame([row])
    if SUMMARY_PATH.exists():
        summary_df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary updated at {SUMMARY_PATH}")


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
        out = model.ask(prompt, think=_run_think, history=None, max_new_tokens=256) if is_local else ask(prompt, model=model, think=_run_think)
        prediction = parse_response(out["content"])
        expected = str(row.expected).strip().upper()
        is_correct = prediction == expected

        after_think = re.split(r"</think>", out["content"], flags=re.IGNORECASE)[-1].strip()
        print(f"  [{i+1}] pred={prediction} expected={expected} {'✓' if is_correct else '✗'} | gap={abs(row.rating_a - row.rating_b):.2f}")
        print(f"       answer: {after_think!r}")

        results.append({
            "id_a": row.id_a, "id_b": row.id_b,
            "rating_a": row.rating_a, "rating_b": row.rating_b,
            "expected": expected, "prediction": prediction,
            "is_correct": is_correct, "raw_response": out["content"],
        })

    pd.DataFrame(results).to_csv(out_path, index=False)
    correct = sum(r["is_correct"] for r in results)
    print(f"\n{model_name}: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    print(f"Results saved to {out_path}")
    append_summary(model_name, results, timestamp)
    model.unload() if is_local else unload(model)


def main():
    global _run_instruction, _run_think
    examples = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(examples)} pairs  |  avg rating gap: {(examples['rating_a'] - examples['rating_b']).abs().mean():.2f}")

    for label_suffix, think, instruction in RUNS:
        _run_instruction = instruction
        _run_think = think
        print(f"\n{'#'*60}\nRun: {label_suffix}  |  think={think}\n{'#'*60}")
        for model_spec in MODELS:
            name, checkpoint, chat_template = model_spec
            model = LocalModel(name, checkpoint, chat_template=chat_template, enable_thinking=False)
            run_test(model, examples, label_suffix)


if __name__ == "__main__":
    main()
