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
DATASET_RATING  = "humicroedit-rating"
DATASET_PAIRWISE = "humicroedit-rating-derived-pairwise"

RUNS = [
    ("", False),
]
# ─────────────────────────────────────────────────────────────────────────────

BASE      = Path(__file__).parent.parent / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset"
S1_TEST   = BASE / "subtask-1" / "test.csv"
S1_TRAIN  = BASE / "subtask-1" / "train.csv"
S2_TEST   = BASE / "subtask-2" / "test.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "humicroedit"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"

N_PAIRS = 1500  # cap on subtask-2 pairs to evaluate; only items needed by those pairs get rated

# Anchor target ratings on 0-3 scale — picked from train set, not test
ANCHOR_TARGETS = [0.3, 1.5, 2.5]

_run_think: bool = False


def apply_edit(original: str, edit: str) -> str:
    return re.sub(r"<[^/]+/>", edit, original).strip()


def select_anchors(train_df: pd.DataFrame) -> list[dict]:
    anchors = []
    for target in ANCHOR_TARGETS:
        row = train_df.iloc[(train_df["meanGrade"] - target).abs().argsort().iloc[:1]].iloc[0]
        anchors.append(row.to_dict())
    return anchors


def build_anchor_block(anchors: list[dict]) -> str:
    lines = []
    for a in anchors:
        headline = apply_edit(a["original"], a["edit"])
        lines.append(f'- Rating {a["meanGrade"]:.1f}/3: "{headline}"')
    return "\n".join(lines)


def build_prompt(row: dict, anchor_block: str) -> str:
    headline = apply_edit(row["original"], row["edit"])
    return (
        "Rate the funniness of this edited news headline on a scale from 0 to 3.\n\n"
        "Calibration examples:\n"
        f"{anchor_block}\n\n"
        f'Headline: "{headline}"\n\n'
        "Return your rating as <rating>X.X</rating> with one decimal place."
    )


def parse_response(content: str) -> float | None:
    m = re.search(r"<rating>(\d+(?:\.\d+)?)</rating>", content, re.IGNORECASE)
    if m:
        return max(0.0, min(3.0, float(m.group(1))))
    # fallback: last number in range
    nums = re.findall(r"\b([0-3](?:\.\d+)?)\b", content)
    if nums:
        return float(nums[-1])
    return None


def append_summary(model: str, n: int, metric_val: float, dataset: str, timestamp: str) -> None:
    row = {
        "model": model, "timestamp": timestamp,
        "n_examples": n,
        "overall_acc": round(metric_val, 1),
        "unknown_count": 0,
        "dataset": dataset,
    }
    df = pd.DataFrame([row])
    if SUMMARY_PATH.exists():
        df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(SUMMARY_PATH, index=False)


def run_test(model, label_suffix: str) -> None:
    is_local = isinstance(model, LocalModel)
    model_name = (model.name if is_local else model) + label_suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load subtask-2 pairs first, cap to N_PAIRS, collect only the item IDs we need
    s2 = pd.read_csv(S2_TEST)
    s2 = s2[s2["label"] != 0].reset_index(drop=True).iloc[:N_PAIRS]
    needed_ids = set()
    for row in s2.itertuples():
        a, b = row.id.split("-")
        needed_ids.add(a); needed_ids.add(b)

    # Build item lookup from subtask-1: id → {original, edit, meanGrade}
    s1 = pd.read_csv(S1_TEST)
    s1["id"] = s1["id"].astype(str)
    item_lookup = s1[s1["id"].isin(needed_ids)].set_index("id").to_dict("index")

    train_df = pd.read_csv(S1_TRAIN)
    anchors = select_anchors(train_df)
    anchor_block = build_anchor_block(anchors)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  {len(item_lookup)} unique items to rate  ({len(s2)} pairs)")
    print(f"Anchors:\n{anchor_block}")
    print(f"{'='*60}")

    results = []
    for i, (item_id, item) in enumerate(item_lookup.items()):
        prompt = build_prompt(item, anchor_block)
        out = model.ask(prompt, think=_run_think, history=None, max_new_tokens=64) if is_local else ask(prompt, model=model, think=_run_think)
        predicted = parse_response(out["content"])
        actual = item["meanGrade"]

        after_think = re.split(r"</think>", out["content"], flags=re.IGNORECASE)[-1].strip()
        print(f"  [{i+1}] actual={actual:.1f} pred={predicted} | {after_think!r}")

        results.append({
            "id": item_id,
            "headline": apply_edit(item["original"], item["edit"]),
            "actual_rating": actual,
            "predicted_rating": predicted,
            "raw_response": out["content"],
        })

    out_path = RESULTS_DIR / f"{model_name.replace(':', '_')}_rating_{timestamp}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    rated = [r for r in results if r["predicted_rating"] is not None]
    rmse = (sum((r["predicted_rating"] - r["actual_rating"]) ** 2 for r in rated) / len(rated)) ** 0.5
    print(f"\n{model_name} Rating RMSE: {rmse:.4f}  (n={len(rated)}, {len(results)-len(rated)} unparsed)")
    print(f"Results saved to {out_path}")
    append_summary(model_name, len(rated), rmse * 100, DATASET_RATING, timestamp)

    # ── Derived pairwise ──────────────────────────────────────────────────────
    score_map = {r["id"]: r["predicted_rating"] for r in results if r["predicted_rating"] is not None}

    pair_results = []
    for row in s2.itertuples():
        id_a, id_b = row.id.split("-")
        predicted = "A" if score_map.get(id_a, -1) >= score_map.get(id_b, -1) else "B"
        expected  = "A" if str(row.label) == "1" else "B"
        pair_results.append({"expected": expected, "predicted": predicted, "is_correct": predicted == expected})

    correct = sum(r["is_correct"] for r in pair_results)
    n_pairs = len(pair_results)
    print(f"{model_name} Derived pairwise: {correct}/{n_pairs} = {correct/n_pairs*100:.1f}%")
    append_summary(model_name, n_pairs, correct / n_pairs * 100, DATASET_PAIRWISE, timestamp)

    pair_out = RESULTS_DIR / f"{model_name.replace(':', '_')}_rating_derived_pairwise_{timestamp}.csv"
    pd.DataFrame(pair_results).to_csv(pair_out, index=False)

    model.unload() if is_local else unload(model)


def main():
    global _run_think
    for label_suffix, think in RUNS:
        _run_think = think
        print(f"\n{'#'*60}\nRun: '{label_suffix}'  |  think={think}\n{'#'*60}")
        for model_spec in MODELS:
            name, checkpoint, chat_template = model_spec
            model = LocalModel(name, checkpoint, chat_template=chat_template, enable_thinking=False)
            run_test(model, label_suffix)


if __name__ == "__main__":
    main()
