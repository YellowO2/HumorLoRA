# ═══════════════════════════════════════════════════════════════════════════════
# Build Humor Reward Dataset (pairwise format)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sources:
#   HaHackathon (SemEval 2021 Task 7): 1000 jokes rated 0–5 by AMT annotators
#   Jester dataset 3: 140 usable jokes rated -10 to +10 by 54,905 users
#
# Outputs:
#   datasets/reward/train.csv     — HaHa 85% jokes + all Jester (chosen/rejected pairs)
#   datasets/reward/test_haha.csv — HaHa 15% held-out jokes (chosen/rejected pairs)
#
# Format: plain text (no chat template) — simplest valid RewardTrainer format
# NYCC is a separate out-of-domain test (eval script prepends image context)
# ═══════════════════════════════════════════════════════════════════════════════

import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HAHA_PATH      = Path(__file__).parent.parent / "datasets" / "hahackathon" / "rating.csv"
JESTER_RATINGS = Path(__file__).parent.parent / "datasets" / "jester" / "dataset3_ratings.csv"
JESTER_JOKES   = Path(__file__).parent.parent / "datasets" / "jester" / "dataset3JokeSet.csv"
TRAIN_PATH     = Path(__file__).parent.parent / "datasets" / "reward" / "train.csv"
TEST_HAHA_PATH = Path(__file__).parent.parent / "datasets" / "reward" / "test_haha.csv"

HAHA_GAP       = 1.0    # min rating gap on 0–5 scale
JESTER_GAP     = 2.0    # min avg rating gap on -10 to +10 scale
MIN_RATERS     = 100
MAX_HAHA_TRAIN = 3000
HAHA_TEST_FRAC = 0.15
SEED           = 42


def make_pairs(rows: list, gap: float, source: str, max_pairs: int = None) -> list[dict]:
    eligible = []
    for a, b in combinations(rows, 2):
        diff = a[1] - b[1]   # a[1] = rating
        if abs(diff) >= gap:
            funnier, less_funny = (a[0], b[0]) if diff > 0 else (b[0], a[0])
            eligible.append({"chosen": funnier, "rejected": less_funny, "source": source})
    if max_pairs and len(eligible) > max_pairs:
        random.seed(SEED)
        eligible = random.sample(eligible, max_pairs)
    return eligible


def load_haha() -> tuple[list[dict], list[dict]]:
    df = pd.read_csv(HAHA_PATH)
    random.seed(SEED)
    indices = list(range(len(df)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - HAHA_TEST_FRAC))
    train_idx = set(indices[:split])
    test_idx  = set(indices[split:])

    train_rows = [(df.iloc[i]["text"], df.iloc[i]["humor_rating"]) for i in train_idx]
    test_rows  = [(df.iloc[i]["text"], df.iloc[i]["humor_rating"]) for i in test_idx]

    train_pairs = make_pairs(train_rows, HAHA_GAP, "haha", MAX_HAHA_TRAIN)
    test_pairs  = make_pairs(test_rows,  HAHA_GAP, "haha")

    print(f"HaHa train jokes: {len(train_rows)} → {len(train_pairs)} pairs")
    print(f"HaHa test  jokes: {len(test_rows)}  → {len(test_pairs)} pairs")
    return train_pairs, test_pairs


def load_jester_pairs() -> list[dict]:
    raw = pd.read_csv(JESTER_RATINGS, header=None)
    ratings = raw.iloc[:, 1:].values.astype(float)
    ratings[ratings == 99] = np.nan

    avg_ratings  = np.nanmean(ratings, axis=0)
    rater_counts = np.sum(~np.isnan(ratings), axis=0)

    jokes = []
    with open(JESTER_JOKES, encoding="utf-8") as f:
        for line in f:
            jokes.append(line.strip().strip('"'))
    assert len(jokes) == 150

    usable = [(jokes[i], avg_ratings[i]) for i in range(150) if rater_counts[i] >= MIN_RATERS]
    pairs  = make_pairs(usable, JESTER_GAP, "jester")
    print(f"Jester: {len(usable)} usable jokes → {len(pairs)} pairs")
    return pairs


def main():
    haha_train, haha_test = load_haha()
    jester_pairs = load_jester_pairs()

    train = haha_train + jester_pairs
    random.seed(SEED)
    random.shuffle(train)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_csv(TRAIN_PATH, index=False)
    pd.DataFrame(haha_test).to_csv(TEST_HAHA_PATH, index=False)

    print(f"\nTrain: {len(train)} pairs  →  {TRAIN_PATH}")
    print(f"Test:  {len(haha_test)} pairs  →  {TEST_HAHA_PATH}")


if __name__ == "__main__":
    main()
