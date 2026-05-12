import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from interact import LocalModel, ask, unload

CHECKPOINT = str(Path(__file__).parent.parent / "outputs" / "gemma4-e2b-discord" / "checkpoint-4011")

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = [
    LocalModel("gemma4-e2b-discord", CHECKPOINT),
    # "gemma4:e2b", "gemma4:e4b", "qwen3:4b", "qwen3.5:9b",
]
N_EXAMPLES = 5  # total examples to evaluate, None for all (~2600 available)
THINK      = False
# ──────────────────────────────────────────────────────────────────────────────

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "newyorker"
RESULTS_DIR  = Path(__file__).parent.parent / "results"


def build_prompt(row: dict) -> str:
    return (
        "You are voting for which caption is funnier in a funny caption contest.\n\n"
        f"The cartoon shows: {row['image_description']}\n"
        f"Uncanny detail: {row['image_uncanny_description']}\n\n"
        "Two captions have been submitted:\n"
        f"A: {row['caption_a']}\n"
        f"B: {row['caption_b']}\n\n"
        "Which caption do you find funnier? Reply with only A or B."
    )


def parse_response(content: str) -> str:
    import re
    m = re.search(r"<answer>([AB])</answer>", content, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # check last word only — avoids false positives from reasoning text
    last_word = content.strip().split()[-1] if content.strip() else ""
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
        combined = combined.head(N_EXAMPLES)
    return combined


def append_summary(model: str, results: list[dict], timestamp: str) -> None:
    summary_path = RESULTS_DIR / "summary.csv"
    df = pd.DataFrame(results)
    row = {
        "model": model,
        "timestamp": timestamp,
        "n_examples": len(df),
        "overall_acc": round(df["is_correct"].mean() * 100, 1),
        "unknown_count": int((df["prediction"] == "UNKNOWN").sum()),
    }
    summary_df = pd.DataFrame([row])
    if summary_path.exists():
        summary_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)
    print(f"Summary updated at {summary_path}")


def run_test(model, examples: pd.DataFrame) -> None:
    is_local = isinstance(model, LocalModel)
    model_name = model.name if is_local else model

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{model_name.replace(':', '_')}_{timestamp}.csv"

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  {len(examples)} examples")
    print(f"{'='*60}")

    results = []
    for i, row in enumerate(examples.itertuples()):
        prompt = build_prompt(row._asdict())
        out = model.ask(prompt, think=THINK) if is_local else ask(prompt, model=model, think=THINK)
        prediction = parse_response(out["content"])
        expected = str(row.expected).strip().upper()
        is_correct = prediction == expected

        print(f"  [{i+1}] fold={row.fold} pred={prediction} expected={expected} {'✓' if is_correct else '✗'}  raw={out['content']!r}")

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
    examples = load_examples()
    print(f"Loaded {len(examples)} examples across {examples['fold'].nunique()} folds")
    for model in MODELS:
        run_test(model, examples)


if __name__ == "__main__":
    main()
