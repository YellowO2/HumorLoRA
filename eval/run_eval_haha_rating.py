"""
Rating eval: ask model to rate each joke 0-5, compare to human average rating.
Metrics: Spearman correlation, MAE.
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from interact import LocalModel

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = [
    ("hermes-3-8b",         "NousResearch/Hermes-3-Llama-3.1-8B", "chatml"),
    ("discord-hermes-3-8b", "mookiezii/Discord-Hermes-3-8B",      "chatml"),
    ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct", "llama-3.1"),
]
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH    = Path(__file__).parent.parent / "datasets" / "hahackathon" / "rating.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "hahackathon"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"


INSTRUCTION = (
    "Rate how funny this text is on a scale of 0 to 5, where:\n"
    "0 = you recognize it's trying to be funny but don't find it funny at all\n"
    "5 = you find it hilarious\n"
    "Return only your rating as <rating>X</rating> where X is a number from 0 to 5."
)


def build_prompt(text: str) -> str:
    return f"{INSTRUCTION}\n\nText: {text}"


def parse_rating(content: str) -> float | None:
    m = re.search(r"<rating>\s*([0-5](?:\.\d+)?)\s*</rating>", content, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # fallback: find any standalone number 0-5 at end of response
    nums = re.findall(r"\b([0-5](?:\.\d+)?)\b", content)
    if nums:
        return float(nums[-1])
    return None


def append_summary(model_name: str, spearman: float, mae: float, n: int, timestamp: str) -> None:
    row = {
        "model": model_name, "timestamp": timestamp,
        "n_examples": n,
        "overall_acc": round(spearman, 4),   # repurpose field: store spearman r
        "unknown_count": 0,
        "dataset": "haha-rating",
    }
    summary_df = pd.DataFrame([row])
    if SUMMARY_PATH.exists():
        summary_df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary: spearman={spearman:.3f}  MAE={mae:.3f}  (n={n})")


def run_test(model: LocalModel, examples: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{model.name}-rating_{timestamp}.csv"

    print(f"\n{'='*60}")
    print(f"Model: {model.name}  |  {len(examples)} texts")
    print(f"{'='*60}")

    results = []
    for i, row in enumerate(examples.itertuples()):
        prompt = build_prompt(row.text)
        out = model.ask(prompt, think=False, history=None, max_new_tokens=64)
        predicted = parse_rating(out["content"])
        print(f"  [{i+1}] human={row.humor_rating:.2f}  model={predicted}  raw={out['content'][:80]!r}")
        results.append({
            "id": row.id, "text": row.text,
            "humor_rating": row.humor_rating,
            "predicted_rating": predicted,
            "raw_response": out["content"],
        })

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    valid = df.dropna(subset=["predicted_rating"])
    if len(valid) < 10:
        print(f"WARNING: only {len(valid)} valid ratings — check parse_rating()")
        return

    spearman, _ = spearmanr(valid["humor_rating"], valid["predicted_rating"])
    mae = (valid["humor_rating"] - valid["predicted_rating"]).abs().mean()
    null_count = df["predicted_rating"].isna().sum()

    print(f"\n{model.name}: Spearman r={spearman:.3f}  MAE={mae:.3f}  unparseable={null_count}")
    append_summary(model.name, spearman, mae, len(valid), timestamp)
    model.unload()


def main():
    examples = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(examples)} texts  |  human rating mean={examples['humor_rating'].mean():.2f}")
    for name, checkpoint, chat_template in MODELS:
        model = LocalModel(name, checkpoint, chat_template=chat_template, enable_thinking=False)
        run_test(model, examples)


if __name__ == "__main__":
    main()
