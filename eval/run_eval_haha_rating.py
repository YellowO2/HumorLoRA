"""
Rating eval: ask model to rate each joke 0-5, compare to human average rating.
Metrics: RMSE, Spearman correlation, MAE.
Also derives pairwise accuracy from predicted scores (higher score wins).
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from interact import LocalModel, ask, unload

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = [
    ("qwen3.5:4b", "unsloth/Qwen3.5-4B", "qwen-3"),
    # previous runs (zero-shot, no anchors):
    # ("hermes-3-8b",         "NousResearch/Hermes-3-Llama-3.1-8B", "chatml"),       # spearman r=0.228
    # ("discord-hermes-3-8b", "mookiezii/Discord-Hermes-3-8B",      "chatml"),       # spearman r=0.015
    # ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct", "llama-3.1"),   # spearman r=0.277
]
DATASET_RATING   = "haha-rating"
DATASET_PAIRWISE = "haha-rating-derived-pairwise"
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH     = Path(__file__).parent.parent / "datasets" / "hahackathon" / "rating.csv"
PAIRWISE_PATH = Path(__file__).parent.parent / "datasets" / "hahackathon" / "pairwise.csv"
RESULTS_DIR   = Path(__file__).parent.parent / "results" / "hahackathon"
SUMMARY_PATH  = Path(__file__).parent.parent / "results" / "summary.csv"

# Anchor targets — held out from test set (no separate train available)
# Realistic crowd average range is ~1.5–4.0 (individual ratings are 1–5)
ANCHOR_TARGETS = [0.2, 1.1, 2.0, 2.9, 4.0]


def select_anchors(df: pd.DataFrame) -> list[dict]:
    anchors = []
    for target in ANCHOR_TARGETS:
        row = df.iloc[(df["humor_rating"] - target).abs().argsort().iloc[:1]].iloc[0]
        anchors.append(row.to_dict())
    return anchors


def build_anchor_block(anchors: list[dict]) -> str:
    lines = []
    for a in anchors:
        lines.append(f'- Rating {a["humor_rating"]:.1f}/5: "{a["text"]}"')
    return "\n".join(lines)


def build_prompt(text: str, anchor_block: str) -> str:
    return (
        "Rate the funniness of this joke on a scale from 0 to 5.\n\n"
        "Calibration examples:\n"
        f"{anchor_block}\n\n"
        f'Joke: "{text}"\n\n'
        "Return your rating as <rating>X.X</rating> with one decimal place."
    )


def parse_rating(content: str) -> float | None:
    m = re.search(r"<rating>\s*([0-5](?:\.\d+)?)\s*</rating>", content, re.IGNORECASE)
    if m:
        return max(0.0, min(5.0, float(m.group(1))))
    nums = re.findall(r"\b([0-5](?:\.\d+)?)\b", content)
    if nums:
        return float(nums[-1])
    return None


def append_summary(model: str, n: int, metric_val: float, dataset: str, timestamp: str) -> None:
    row = {
        "model": model, "timestamp": timestamp,
        "n_examples": n,
        "overall_acc": round(metric_val, 4),
        "unknown_count": 0,
        "dataset": dataset,
    }
    pd.DataFrame([row]).to_csv(
        SUMMARY_PATH, mode="a",
        header=not SUMMARY_PATH.exists(), index=False
    )


def run_test(model, examples: pd.DataFrame) -> None:
    is_local = isinstance(model, LocalModel)
    model_name = model.name if is_local else model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    anchors = select_anchors(examples)
    anchor_ids = {str(a["id"]) for a in anchors}
    eval_df = examples[~examples["id"].astype(str).isin(anchor_ids)].reset_index(drop=True)
    anchor_block = build_anchor_block(anchors)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  {len(eval_df)} jokes  ({len(anchors)} held out as anchors)")
    print(f"Anchors:\n{anchor_block}")
    print(f"{'='*60}")

    results = []
    for i, row in enumerate(eval_df.itertuples()):
        prompt = build_prompt(row.text, anchor_block)
        out = model.ask(prompt, think=False, history=None, max_new_tokens=64) if is_local else ask(prompt, model=model, think=False)
        predicted = parse_rating(out["content"])
        after_think = re.split(r"</think>", out["content"], flags=re.IGNORECASE)[-1].strip()
        print(f"  [{i+1}] human={row.humor_rating:.2f}  model={predicted}  {after_think!r}")
        results.append({
            "id": row.id, "text": row.text,
            "humor_rating": row.humor_rating,
            "predicted_rating": predicted,
            "raw_response": out["content"],
        })

    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / f"{model_name.replace(':', '_')}_rating_{timestamp}.csv"
    df.to_csv(out_path, index=False)

    valid = df.dropna(subset=["predicted_rating"])
    if len(valid) < 10:
        print(f"WARNING: only {len(valid)} valid ratings — check parse_rating()")
        return

    rmse = ((valid["humor_rating"] - valid["predicted_rating"]) ** 2).mean() ** 0.5
    spearman, _ = spearmanr(valid["humor_rating"], valid["predicted_rating"])
    mae = (valid["humor_rating"] - valid["predicted_rating"]).abs().mean()
    print(f"\n{model_name}: RMSE={rmse:.4f}  Spearman r={spearman:.3f}  MAE={mae:.3f}  unparsed={df['predicted_rating'].isna().sum()}")
    print(f"Results saved to {out_path}")
    append_summary(model_name, len(valid), rmse * 100, DATASET_RATING, timestamp)

    # ── Derived pairwise ──────────────────────────────────────────────────────
    score_map = dict(zip(df["id"].astype(str), df["predicted_rating"]))
    pairs_df = pd.read_csv(PAIRWISE_PATH)

    pair_results = []
    skipped = 0
    for row in pairs_df.itertuples():
        id_a, id_b = str(row.id_a), str(row.id_b)
        if score_map.get(id_a) is None or score_map.get(id_b) is None:
            skipped += 1
            continue
        predicted = "A" if score_map[id_a] >= score_map[id_b] else "B"
        expected  = str(row.expected).strip().upper()
        pair_results.append({"expected": expected, "predicted": predicted, "is_correct": predicted == expected})

    correct = sum(r["is_correct"] for r in pair_results)
    n_pairs = len(pair_results)
    acc = correct / n_pairs * 100
    print(f"{model_name} Derived pairwise: {correct}/{n_pairs} = {acc:.1f}%  (skipped {skipped} anchor/unparsed pairs)")
    append_summary(model_name, n_pairs, acc, DATASET_PAIRWISE, timestamp)

    pair_out = RESULTS_DIR / f"{model_name.replace(':', '_')}_rating_derived_pairwise_{timestamp}.csv"
    pd.DataFrame(pair_results).to_csv(pair_out, index=False)

    model.unload() if is_local else unload(model)


def main():
    examples = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(examples)} jokes  |  human rating mean={examples['humor_rating'].mean():.2f}")
    for model_spec in MODELS:
        name, checkpoint, chat_template = model_spec
        model = LocalModel(name, checkpoint, chat_template=chat_template, enable_thinking=False)
        run_test(model, examples)


if __name__ == "__main__":
    main()
