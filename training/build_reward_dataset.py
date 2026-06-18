# ═══════════════════════════════════════════════════════════════════════════════
# Build Humor Reward Dataset
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sources:
#   HaHackathon (SemEval 2021 Task 7): 1000 jokes rated 0–5 by AMT annotators
#   Jester dataset 3: 140 usable jokes rated -10 to +10 by 54,905 users
#
# Output: datasets/reward/train.csv  (columns: text, score, source)
#   score normalized to [0, 1] for both datasets
#
# Test set is NYCC pairwise (held out entirely — different domain, different crowd)
# ═══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
import numpy as np
import pandas as pd

HAHA_PATH    = Path(__file__).parent.parent / "datasets" / "hahackathon" / "rating.csv"
JESTER_RATINGS = Path(__file__).parent.parent / "datasets" / "jester" / "dataset3_ratings.csv"
JESTER_JOKES   = Path(__file__).parent.parent / "datasets" / "jester" / "dataset3JokeSet.csv"
OUTPUT_PATH  = Path(__file__).parent.parent / "datasets" / "reward" / "train.csv"
MIN_RATERS   = 100


# ── HaHackathon ───────────────────────────────────────────────────────────────

def load_haha() -> pd.DataFrame:
    df = pd.read_csv(HAHA_PATH)
    # humor_rating is 0–5; normalize to [0, 1]
    df["score"] = df["humor_rating"] / 5.0
    df["source"] = "haha"
    return df[["text", "score", "source"]]


# ── Jester ────────────────────────────────────────────────────────────────────

def load_jester() -> pd.DataFrame:
    # ratings matrix: (n_users, 151) — first col = count of jokes rated per user
    raw = pd.read_csv(JESTER_RATINGS, header=None)
    ratings = raw.iloc[:, 1:].values.astype(float)   # (54905, 150)
    ratings[ratings == 99] = np.nan

    # per-joke avg and rater count
    avg_ratings  = np.nanmean(ratings, axis=0)   # (150,)
    rater_counts = np.sum(~np.isnan(ratings), axis=0)

    jokes = []
    with open(JESTER_JOKES, encoding="utf-8") as f:
        for line in f:
            jokes.append(line.strip().strip('"'))
    assert len(jokes) == 150

    rows = []
    for i, (text, avg, count) in enumerate(zip(jokes, avg_ratings, rater_counts)):
        if count < MIN_RATERS:
            continue
        # normalize -10..+10 → 0..1
        score = (avg + 10.0) / 20.0
        rows.append({"text": text, "score": score, "source": "jester"})

    return pd.DataFrame(rows)


# ── Combine & save ────────────────────────────────────────────────────────────

def main():
    haha   = load_haha()
    jester = load_jester()

    combined = pd.concat([haha, jester], ignore_index=True)
    combined = combined.dropna(subset=["text", "score"])
    combined = combined[combined["text"].str.strip() != ""]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"HaHa:   {len(haha)} jokes  |  score range [{haha['score'].min():.3f}, {haha['score'].max():.3f}]  |  mean {haha['score'].mean():.3f}")
    print(f"Jester: {len(jester)} jokes  |  score range [{jester['score'].min():.3f}, {jester['score'].max():.3f}]  |  mean {jester['score'].mean():.3f}")
    print(f"Total:  {len(combined)} jokes  →  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
