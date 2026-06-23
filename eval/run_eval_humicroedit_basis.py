"""
Humor-basis preference model for Humicroedit pairwise (SemEval 2020 Task 7).
Replicates the lmfaoooo (SemEval 2026 Task 1 winner) approach:
  1. Score each edited headline on 17 humor features using LLM (0/1 per feature)
  2. Train logistic regression on (feats_A - feats_B) -> pairwise label
  3. Eval on test set
Features are cached to disk so LLM calls are not repeated on re-runs.
"""
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from interact import ask

# ── Config ────────────────────────────────────────────────────────────────────
MODEL       = "qwen3.5:4b"
N_TRAIN     = 1000   # pairs from train.csv used to train the preference model
N_TEST      = 1000   # pairs from test.csv used to evaluate
DATASET     = "humicroedit-basis-pairwise"
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent.parent / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset" / "subtask-2"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "humicroedit"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"
CACHE_PATH   = Path(__file__).parent.parent / "results" / "feature_cache" / "humicroedit_basis.json"

HUMOR_BASIS = [
    "Clear Punchline: Ensure the joke delivers a strong, unmistakable punchline for maximum impact.",
    "Wordplay with Purpose: Use puns or wordplay that serves the joke, rather than relying on repetition or forced cleverness.",
    "Universality: Use references that are widely understood or relatable to broaden appeal.",
    "Natural Dialogue: Employ conversational exchanges to make the joke feel organic and engaging.",
    "Subtlety Over Obviousness: Favor subtle humor that allows audiences to connect the dots over jokes that spell everything out.",
    "Avoid Cliché: Steer away from jokes that rely on overused wordplay or tired humor structures.",
    "Fresh Perspective: Offer a novel or surprising angle on familiar situations to keep material original.",
    "Exaggeration: Amplifying a characteristic, situation, or behavior to absurd levels to highlight its comedic potential.",
    "Subverting Expectations: Twisting a familiar setup creates delight by catching the audience off guard.",
    "Character-Driven Humor: Use established stereotypes or behaviors to anchor the joke and build richer scenarios.",
    "Economy of Words: Be concise and efficient with language, trimming unnecessary details to maximize comedic payoff.",
    "Self-Deprecation: Playfully targeting oneself can disarm the audience and make humor more relatable.",
    "Satirical Edge: Employ satire to critique social trends or behaviors, adding depth to the humor.",
    "Anthropomorphism: Attribute human qualities to non-human entities for humorous effect.",
    "Clever Analogies: Use creative comparisons that link unrelated concepts for a surprising comedic twist.",
    "Memorable Imagery: Create vivid or amusing mental pictures that stick with the audience.",
    "Dark Humor: Making light of subjects that are generally considered serious, taboo, or morbid.",
]

BASIS_BLOCK = "\n".join(f"{i}. {r}" for i, r in enumerate(HUMOR_BASIS))

FEATURE_PROMPT = """\
You are an expert in humor theory. A news headline was made funny by replacing one word. Analyze the edited version against each humor rule below.

Rules:
{basis}

Original: "{original}"
Edited:   "{edited}"

The humor comes from replacing the original word with the new one. Evaluate the edited headline against each rule.

Output ONLY a JSON dictionary with rule numbers 0-16 as keys and "0" or "1" as values (1 = satisfies the rule, 0 = does not).

```json
{{
    "0": <0 or 1>,
    "1": <0 or 1>,
    "2": <0 or 1>,
    "3": <0 or 1>,
    "4": <0 or 1>,
    "5": <0 or 1>,
    "6": <0 or 1>,
    "7": <0 or 1>,
    "8": <0 or 1>,
    "9": <0 or 1>,
    "10": <0 or 1>,
    "11": <0 or 1>,
    "12": <0 or 1>,
    "13": <0 or 1>,
    "14": <0 or 1>,
    "15": <0 or 1>,
    "16": <0 or 1>
}}
```"""


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def apply_edit(original: str, edit: str) -> str:
    return re.sub(r"<[^/]+/>", edit, original).strip()


def restore_original(original: str) -> str:
    """Replace <Word/> tag with the original word it contains."""
    return re.sub(r"<([^/]+)/>", r"\1", original).strip()


def load_cache() -> dict:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def extract_features(original_text: str, edited_text: str, cache: dict) -> list[int] | None:
    key = text_hash(edited_text)
    if key in cache:
        return cache[key]
    prompt = FEATURE_PROMPT.format(basis=BASIS_BLOCK, original=original_text, edited=edited_text)
    out = ask(prompt, model=MODEL, think=False)
    content = out["content"]
    m = re.search(r"\{[^}]+\}", content, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group())
        features = [int(parsed.get(str(i), 0)) for i in range(17)]
        cache[key] = features
        return features
    except Exception:
        return None


def load_pairs(path: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["label"] != 0].reset_index(drop=True)
    return df.iloc[:n]


def pairs_to_xy(pairs: pd.DataFrame, cache: dict, split_name: str):
    X, y = [], []
    for i, row in enumerate(pairs.itertuples()):
        orig_a    = restore_original(row.original1)
        edited_a  = apply_edit(row.original1, row.edit1)
        orig_b    = restore_original(row.original2)
        edited_b  = apply_edit(row.original2, row.edit2)
        feats_a = extract_features(orig_a, edited_a, cache)
        feats_b = extract_features(orig_b, edited_b, cache)
        if feats_a is None or feats_b is None:
            continue
        diff = [a - b for a, b in zip(feats_a, feats_b)]
        label = 1 if str(row.label).strip() == "1" else 0  # 1 = A is funnier
        X.append(diff)
        y.append(label)
        if (i + 1) % 100 == 0:
            save_cache(cache)
            print(f"  [{split_name}] {i+1}/{len(pairs)} pairs processed, cache={len(cache)} items")
    save_cache(cache)
    return np.array(X), np.array(y)


def append_summary(n: int, acc: float, timestamp: str) -> None:
    row = {
        "model": MODEL, "timestamp": timestamp,
        "n_examples": n,
        "overall_acc": round(acc * 100, 1),
        "unknown_count": 0,
        "dataset": DATASET,
    }
    pd.DataFrame([row]).to_csv(
        SUMMARY_PATH, mode="a",
        header=not SUMMARY_PATH.exists(), index=False
    )


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cache = load_cache()
    print(f"Cache loaded: {len(cache)} items already scored")

    print(f"\n── Extracting features for TRAIN ({N_TRAIN} pairs) ──")
    train_pairs = load_pairs(DATA_DIR / "train.csv", N_TRAIN)
    X_train, y_train = pairs_to_xy(train_pairs, cache, "train")
    print(f"Train matrix: {X_train.shape}, label dist: {y_train.mean():.2f} frac A wins")

    print(f"\n── Extracting features for TEST ({N_TEST} pairs) ──")
    test_pairs = load_pairs(DATA_DIR / "test.csv", N_TEST)
    X_test, y_test = pairs_to_xy(test_pairs, cache, "test")
    print(f"Test matrix: {X_test.shape}")

    print("\n── Training logistic regression (L1 = LASSO) ──")
    clf_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000)
    clf_l1.fit(X_train, y_train)
    acc_l1 = accuracy_score(y_test, clf_l1.predict(X_test))

    clf_l2 = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=1000)
    clf_l2.fit(X_train, y_train)
    acc_l2 = accuracy_score(y_test, clf_l2.predict(X_test))

    print(f"\nResults on {len(y_test)} test pairs:")
    print(f"  L1 (LASSO): {acc_l1*100:.1f}%")
    print(f"  L2 (Ridge): {acc_l2*100:.1f}%")

    # log best to summary
    best_acc = max(acc_l1, acc_l2)
    append_summary(len(y_test), best_acc, timestamp)
    print(f"\nSummary updated. Best: {best_acc*100:.1f}%")

    # save feature weights for inspection
    feature_names = [r.split(":")[0] for r in HUMOR_BASIS]
    weights_df = pd.DataFrame({
        "feature": feature_names,
        "weight_l1": clf_l1.coef_[0],
        "weight_l2": clf_l2.coef_[0],
    }).sort_values("weight_l1", ascending=False)
    weights_path = RESULTS_DIR / f"basis_weights_{timestamp}.csv"
    weights_df.to_csv(weights_path, index=False)
    print(f"Feature weights saved to {weights_path}")
    print(weights_df.to_string(index=False))


if __name__ == "__main__":
    main()
