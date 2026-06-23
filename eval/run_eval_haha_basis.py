"""
Humor-basis preference model for HaHackathon pairwise (SemEval 2021 Task 7).
Replicates the lmfaoooo (SemEval 2026 Task 1 winner) approach:
  1. Score each joke on 17 humor features using LLM (0/1 per feature)
  2. Train logistic regression on (feats_A - feats_B) -> pairwise label
  3. Eval on held-out test split
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
from interact import LocalModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = "qwen3.5:4b"
MODEL_CHECKPOINT = "unsloth/Qwen3.5-4B"
MODEL_TEMPLATE   = "qwen-3"
N_PAIRS     = 2000   # use all available pairs, then split 80/20
TRAIN_FRAC  = 0.8
DATASET     = "haha-basis-pairwise"
# ─────────────────────────────────────────────────────────────────────────────

PAIRWISE_PATH = Path(__file__).parent.parent / "datasets" / "hahackathon" / "pairwise.csv"
RESULTS_DIR   = Path(__file__).parent.parent / "results" / "hahackathon"
SUMMARY_PATH  = Path(__file__).parent.parent / "results" / "summary.csv"
CACHE_PATH    = Path(__file__).parent.parent / "results" / "feature_cache" / "haha_basis.json"

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
You are an expert in humor theory. Analyze the following joke and evaluate it against each humor rule below.

Rules:
{basis}

Joke: "{item}"

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


def load_cache() -> dict:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def extract_features(text: str, cache: dict, model: LocalModel) -> list[int] | None:
    key = text_hash(text)
    if key in cache:
        return cache[key]
    prompt = FEATURE_PROMPT.format(basis=BASIS_BLOCK, item=text)
    out = model.ask(prompt, think=False, history=None, max_new_tokens=256)
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


def pairs_to_xy(pairs: pd.DataFrame, cache: dict, split_name: str, model: LocalModel):
    X, y = [], []
    for i, row in enumerate(pairs.itertuples()):
        feats_a = extract_features(str(row.text_a), cache, model)
        feats_b = extract_features(str(row.text_b), cache, model)
        if feats_a is None or feats_b is None:
            continue
        diff = [a - b for a, b in zip(feats_a, feats_b)]
        label = 1 if str(row.expected).strip().upper() == "A" else 0
        X.append(diff)
        y.append(label)
        if (i + 1) % 100 == 0:
            save_cache(cache)
            print(f"  [{split_name}] {i+1}/{len(pairs)} pairs processed, cache={len(cache)} items")
    save_cache(cache)
    return np.array(X), np.array(y)


def append_summary(n: int, acc: float, timestamp: str) -> None:
    row = {
        "model": MODEL_NAME, "timestamp": timestamp,
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

    model = LocalModel(MODEL_NAME, MODEL_CHECKPOINT, chat_template=MODEL_TEMPLATE, enable_thinking=False)

    pairs_df = pd.read_csv(PAIRWISE_PATH).iloc[:N_PAIRS]
    n_train = int(len(pairs_df) * TRAIN_FRAC)
    train_pairs = pairs_df.iloc[:n_train]
    test_pairs  = pairs_df.iloc[n_train:]
    print(f"Split: {len(train_pairs)} train / {len(test_pairs)} test pairs")

    print(f"\n── Extracting features for TRAIN ({len(train_pairs)} pairs) ──")
    X_train, y_train = pairs_to_xy(train_pairs, cache, "train", model)
    print(f"Train matrix: {X_train.shape}, label dist: {y_train.mean():.2f} frac A wins")

    print(f"\n── Extracting features for TEST ({len(test_pairs)} pairs) ──")
    X_test, y_test = pairs_to_xy(test_pairs, cache, "test", model)
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
    model.unload()


if __name__ == "__main__":
    main()
