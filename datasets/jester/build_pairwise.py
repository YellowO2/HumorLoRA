"""
Build pairwise CSV from Jester dataset 3.

Ratings matrix: 54,905 users x 150 jokes, 99 = not rated, scale -10 to +10.
Outputs: datasets/jester/pairwise.csv

Columns: id_a, id_b, text_a, text_b, rating_a, rating_b, expected
  - id_a/id_b: 1-indexed joke IDs
  - expected: A if avg(joke_a) > avg(joke_b), else B
  - only pairs with |avg_a - avg_b| > GAP_THRESHOLD are included
"""

import argparse
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RATINGS_PATH  = Path(__file__).parent / "dataset3_ratings.csv"
JOKES_PATH    = Path(__file__).parent / "dataset3JokeSet.csv"
OUTPUT_PATH   = Path(__file__).parent / "pairwise.csv"

GAP_THRESHOLD   = 2.0
MIN_RATERS      = 100    # drop jokes with fewer raters than this
MAX_PAIRS       = 2000
RANDOM_SEED     = 42


def load_ratings() -> np.ndarray:
    """Return (n_users, 150) float array; NaN where unrated."""
    df = pd.read_csv(RATINGS_PATH, header=None)
    # first column = count of jokes rated by that user; drop it
    ratings = df.iloc[:, 1:].values.astype(float)
    ratings[ratings == 99] = np.nan
    return ratings  # shape: (54905, 150)


def load_jokes() -> list[str]:
    """Return list of 150 joke texts (0-indexed)."""
    jokes = []
    with open(JOKES_PATH, encoding="utf-8") as f:
        for line in f:
            text = line.strip().strip('"')
            jokes.append(text)
    return jokes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap",      type=float, default=GAP_THRESHOLD)
    parser.add_argument("--min-raters", type=int, default=MIN_RATERS)
    parser.add_argument("--max-pairs",  type=int, default=MAX_PAIRS)
    parser.add_argument("--seed",     type=int,   default=RANDOM_SEED)
    parser.add_argument("--out",      default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Loading ratings matrix...")
    ratings = load_ratings()
    print(f"  Shape: {ratings.shape}")

    jokes = load_jokes()
    assert len(jokes) == 150, f"Expected 150 jokes, got {len(jokes)}"

    # per-joke stats
    rater_counts = np.sum(~np.isnan(ratings), axis=0)   # (150,)
    avg_ratings  = np.nanmean(ratings, axis=0)           # (150,)

    usable_ids = [i for i in range(150) if rater_counts[i] >= args.min_raters]
    print(f"  Usable jokes (≥{args.min_raters} raters): {len(usable_ids)}")
    print(f"  Avg rating range: {avg_ratings[usable_ids].min():.2f} to {avg_ratings[usable_ids].max():.2f}")

    # all pairs among usable jokes with gap > threshold
    eligible = []
    for i, j in combinations(usable_ids, 2):
        diff = avg_ratings[i] - avg_ratings[j]
        if abs(diff) > args.gap:
            eligible.append((i, j, avg_ratings[i], avg_ratings[j]))

    print(f"  Pairs with gap > {args.gap}: {len(eligible)}")

    random.seed(args.seed)
    sampled = random.sample(eligible, min(args.max_pairs, len(eligible)))
    print(f"  Sampled: {len(sampled)}")

    rows = []
    for i, j, ra, rj in sampled:
        # ensure A is always the higher-rated one (expected = A)
        if ra >= rj:
            id_a, id_b, text_a, text_b, rating_a, rating_b = i+1, j+1, jokes[i], jokes[j], ra, rj
        else:
            id_a, id_b, text_a, text_b, rating_a, rating_b = j+1, i+1, jokes[j], jokes[i], rj, ra
        rows.append({
            "id_a": id_a, "id_b": id_b,
            "text_a": text_a, "text_b": text_b,
            "rating_a": round(rating_a, 4),
            "rating_b": round(rating_b, 4),
            "expected": "A",
        })

    # shuffle so A isn't always the answer
    random.shuffle(rows)
    for idx, row in enumerate(rows):
        if idx % 2 == 1:
            row["id_a"], row["id_b"] = row["id_b"], row["id_a"]
            row["text_a"], row["text_b"] = row["text_b"], row["text_a"]
            row["rating_a"], row["rating_b"] = row["rating_b"], row["rating_a"]
            row["expected"] = "B"

    df_out = pd.DataFrame(rows, columns=["id_a", "id_b", "text_a", "text_b", "rating_a", "rating_b", "expected"])
    df_out.to_csv(args.out, index=False)

    avg_gap = (df_out["rating_a"] - df_out["rating_b"]).abs().mean()
    a_frac  = (df_out["expected"] == "A").mean()
    print(f"\nSaved {len(df_out)} pairs to {args.out}")
    print(f"  Avg |gap|: {avg_gap:.2f}")
    print(f"  A/B balance: {a_frac:.1%} / {1-a_frac:.1%}")


if __name__ == "__main__":
    main()
