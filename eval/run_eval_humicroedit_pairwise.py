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
    ("qwen3.5:4b", "unsloth/Qwen3.5-4B", "qwen-3"),
]
DATASET = "humicroedit-pairwise"

RUNS = [
    ("-no-gut", False, "Return only <answer>A</answer> or <answer>B</answer>."),
]
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH    = Path(__file__).parent.parent / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset" / "subtask-2" / "test.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "humicroedit"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"

_run_instruction: str = ""
_run_think: bool = False


def apply_edit(original: str, edit: str) -> str:
    return re.sub(r"<[^/]+/>", edit, original).strip()


def build_prompt(row: dict) -> str:
    headline_a = apply_edit(row["original1"], row["edit1"])
    headline_b = apply_edit(row["original2"], row["edit2"])
    return (
        "You are judging which of two edited news headlines is funnier.\n\n"
        f"A: {headline_a}\n"
        f"B: {headline_b}\n\n"
        f"Which headline is funnier? {_run_instruction}"
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
        # label is 1 or 2 — map to A/B
        expected = "A" if str(row.label).strip() == "1" else "B"
        is_correct = prediction == expected

        after_think = re.split(r"</think>", out["content"], flags=re.IGNORECASE)[-1].strip()
        print(f"  [{i+1}] pred={prediction} expected={expected} {'✓' if is_correct else '✗'}")
        print(f"       answer: {after_think!r}")

        results.append({
            "id": row.id,
            "edit1": row.edit1, "edit2": row.edit2,
            "mean_grade1": row.meanGrade1, "mean_grade2": row.meanGrade2,
            "label": row.label,
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
    print(f"Loaded {len(examples)} pairs")
    print(f"Label distribution: {examples['label'].value_counts().to_dict()}")
    ties = (examples["label"] == 0).sum()
    examples = examples[examples["label"] != 0].reset_index(drop=True)
    print(f"Excluded {ties} ties (label=0), evaluating on {len(examples)} pairs")

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
