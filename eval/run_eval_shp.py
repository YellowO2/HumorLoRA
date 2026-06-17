import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from interact import LocalModel, ask, unload

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = [
    # ── Ollama baselines ──────────────────────────────────────────────────────
    # "qwen3.5:4b",
    # ── HF baselines ─────────────────────────────────────────────────────────
    # ("qwen3.5:4b", "unsloth/Qwen3.5-4B", "qwen-3"),
    # ("hermes-3-8b", "NousResearch/Hermes-3-Llama-3.1-8B", "chatml"),
    # ── Fine-tuned checkpoints ────────────────────────────────────────────────
    ("discord-hermes-3-8b", "mookiezii/Discord-Hermes-3-8B", "chatml"),
]
N_EXAMPLES = 2000  # None for all

RUNS = [
    # gut == no-gut confirmed — plain only going forward
    ("-no-gut", False, "Return <answer>A</answer> or <answer>B</answer>."),
]
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results" / "shp"

_run_instruction: str = ""
_run_think: bool = False


def build_prompt(row: dict) -> str:
    return (
        "You are a reddit user reading this post. Judge which of two Reddit comments you'd like to upvote.\n\n"
        f"Post:\n{row['history']}\n\n"
        f"Comment A:\n{row['human_ref_A']}\n\n"
        f"Comment B:\n{row['human_ref_B']}\n\n"
        f"{_run_instruction}"
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
    import requests, json, time
    subreddits = [
        "askacademia", "askanthropology", "askbaking", "askcarguys",
        "askculinary", "askdocs", "askengineers", "askhistorians",
        "askhr", "askphysics", "askscience", "asksocialscience",
        "askwomenadvice", "eli5", "explainlikeimfive", "legaladvice",
        "personalfinance", "relationships",
    ]
    base = "https://huggingface.co/datasets/stanfordnlp/SHP/resolve/main"
    rows = []
    for sub in subreddits:
        url = f"{base}/{sub}/validation.json"
        for attempt in range(5):
            try:
                r = requests.get(url, timeout=300)
                break
            except requests.exceptions.Timeout:
                print(f"  timeout on {sub} (attempt {attempt+1}/5), retrying...")
                time.sleep(5)
        else:
            print(f"  skipping {sub} after 5 timeouts")
            continue
        if r.status_code != 200:
            print(f"  skipping {sub} ({r.status_code})")
            continue
        for line in r.text.strip().split("\n"):
            if line:
                rows.append(json.loads(line))
        print(f"  loaded {sub} ({len(rows)} total so far)")
        if N_EXAMPLES and len(rows) >= N_EXAMPLES * 3:
            break
    df = pd.DataFrame(rows)
    df = df[df["score_ratio"] >= 2].reset_index(drop=True)
    if N_EXAMPLES:
        df = df.head(N_EXAMPLES)
    print(f"Loaded {len(df)} SHP examples (score_ratio >= 2)")
    return df


SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"

def append_summary(model_name: str, results: list[dict], timestamp: str) -> None:
    df = pd.DataFrame(results)
    row = {
        "model": model_name,
        "timestamp": timestamp,
        "n_examples": len(df),
        "overall_acc": round(df["is_correct"].mean() * 100, 1),
        "unknown_count": int((df["prediction"] == "UNKNOWN").sum()),
        "dataset": "shp",
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
        out = model.ask(prompt, think=_run_think, history=None, max_new_tokens=512) if is_local else ask(prompt, model=model, think=_run_think)
        prediction = parse_response(out["content"])
        # labels=1 means A is preferred, labels=0 means B is preferred
        expected = "A" if row.labels == 1 else "B"
        is_correct = prediction == expected

        import re as _re
        think_match = _re.search(r"<think>(.*?)</think>", out["content"], _re.DOTALL | _re.IGNORECASE)
        think_text = think_match.group(1).strip() if think_match else ""
        after_think = _re.split(r"</think>", out["content"], flags=_re.IGNORECASE)[-1].strip()
        print(f"  [{i+1}] pred={prediction} expected={expected} {'✓' if is_correct else '✗'}")
        print(f"       think({len(think_text)} chars): {think_text[:150].replace(chr(10), ' ')!r}")
        print(f"       answer: {after_think!r}")

        results.append({
            "post_id": row.post_id,
            "domain": row.domain,
            "expected": expected,
            "prediction": prediction,
            "is_correct": is_correct,
            "score_ratio": row.score_ratio,
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
