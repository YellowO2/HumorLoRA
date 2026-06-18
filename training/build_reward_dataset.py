# ═══════════════════════════════════════════════════════════════════════════════
# Build Humor Reward Dataset (pairwise format)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sources:
#   HaHackathon (SemEval 2021 Task 7): 1000 jokes rated 0–5 by AMT annotators
#   Jester dataset 3: 140 usable jokes rated -10 to +10 by 54,905 users
#
# Output: datasets/reward/train.csv  (columns: chosen, rejected, source)
#   chosen  = funnier joke text
#   rejected = less funny joke text
#
# Test set is NYCC pairwise (held out entirely — different domain, different crowd)
# ═══════════════════════════════════════════════════════════════════════════════

import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HAHA_PATH      = Path(__file__).parent.parent / "datasets" / "hahackathon" / "rating.csv"
JESTER_RATINGS = Path(__file__).parent.parent / "datasets" / "jester" / "dataset3_ratings.csv"
JESTER_JOKES   = Path(__file__).parent.parent / "datasets" / "jester" / "dataset3JokeSet.csv"
OUTPUT_PATH    = Path(__file__).parent.parent / "datasets" / "reward" / "train.csv"

HAHA_GAP    = 1.0    # min rating gap on 0–5 scale
JESTER_GAP  = 2.0    # min avg rating gap on -10 to +10 scale
MIN_RATERS  = 100
MAX_HAHA    = 3000
SEED        = 42


def load_haha_pairs() -> list[dict]:
    df = pd.read_csv(HAHA_PATH)
    rows = list(df.itertuples(index=False))
    eligible = []
    for a, b in combinations(rows, 2):
        gap = a.humor_rating - b.humor_rating
        if abs(gap) >= HAHA_GAP:
            chosen, rejected = (a.text, b.text) if gap > 0 else (b.text, a.text)
            eligible.append({"chosen": chosen, "rejected": rejected, "source": "haha"})

    random.seed(SEED)
    sampled = random.sample(eligible, min(MAX_HAHA, len(eligible)))
    print(f"HaHa: {len(eligible)} eligible pairs → sampled {len(sampled)}")
    return sampled


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

    usable = [(i, jokes[i], avg_ratings[i]) for i in range(150) if rater_counts[i] >= MIN_RATERS]

    eligible = []
    for (i, text_i, avg_i), (j, text_j, avg_j) in combinations(usable, 2):
        gap = avg_i - avg_j
        if abs(gap) >= JESTER_GAP:
            chosen, rejected = (text_i, text_j) if gap > 0 else (text_j, text_i)
            eligible.append({"chosen": chosen, "rejected": rejected, "source": "jester"})

    print(f"Jester: {len(eligible)} eligible pairs (all kept)")
    return eligible


def main():
    haha_pairs   = load_haha_pairs()
    jester_pairs = load_jester_pairs()

    combined = haha_pairs + jester_pairs
    random.seed(SEED)
    random.shuffle(combined)

    df_out = pd.DataFrame(combined)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTotal: {len(df_out)} pairs  →  {OUTPUT_PATH}")
    print(f"  HaHa: {sum(1 for r in combined if r['source'] == 'haha')}")
    print(f"  Jester: {sum(1 for r in combined if r['source'] == 'jester')}")


if __name__ == "__main__":
    main()
