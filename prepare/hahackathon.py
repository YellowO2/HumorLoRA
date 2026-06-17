"""
Download HaHackathon train split and produce two eval datasets:
  datasets/hahackathon/pairwise.csv  — A/B pairs for accuracy eval
  datasets/hahackathon/rating.csv    — individual jokes for correlation eval
"""
import random
import itertools
from pathlib import Path

import pandas as pd

TRAIN_URL = (
    "https://raw.githubusercontent.com/NLP-UMUTeam/SemEval2021-HaHackathon-UMUTeam"
    "/main/datasets/hahackathon_train.csv"
)

RATING_DIFF_THRESHOLD = 1.0   # minimum humor_rating gap to form a pair
N_PAIRS  = 2000
N_RATING = 1000
SEED     = 42

OUT_DIR = Path(__file__).parent.parent / "datasets" / "hahackathon"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Downloading HaHackathon train split...")
    df = pd.read_csv(TRAIN_URL)
    print(f"Total rows: {len(df)}")

    humorous = df[df["is_humor"] == 1].reset_index(drop=True)
    print(f"Humorous texts: {len(humorous)}")

    # ── Pairwise dataset ──────────────────────────────────────────────────────
    rng = random.Random(SEED)
    rows = humorous.to_dict("records")

    pairs = []
    # random sampling — check random pairs until we have enough
    indices = list(range(len(rows)))
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < N_PAIRS * 50:
        i, j = rng.sample(indices, 2)
        a, b = rows[i], rows[j]
        diff = abs(a["humor_rating"] - b["humor_rating"])
        if diff >= RATING_DIFF_THRESHOLD:
            # randomly assign which is text_a and which is text_b
            if rng.random() < 0.5:
                a, b = b, a
            winner = "A" if a["humor_rating"] > b["humor_rating"] else "B"
            pairs.append({
                "id_a": a["id"], "id_b": b["id"],
                "text_a": a["text"], "text_b": b["text"],
                "rating_a": a["humor_rating"], "rating_b": b["humor_rating"],
                "expected": winner,
            })
        attempts += 1

    pairwise_df = pd.DataFrame(pairs)
    pairwise_path = OUT_DIR / "pairwise.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"Pairwise: {len(pairwise_df)} pairs saved to {pairwise_path}")
    print(f"  avg rating diff: {(pairwise_df['rating_a'] - pairwise_df['rating_b']).abs().mean():.2f}")

    # ── Rating dataset ────────────────────────────────────────────────────────
    rating_df = humorous.sample(n=N_RATING, random_state=SEED)[["id", "text", "humor_rating"]].reset_index(drop=True)
    rating_path = OUT_DIR / "rating.csv"
    rating_df.to_csv(rating_path, index=False)
    print(f"Rating: {len(rating_df)} texts saved to {rating_path}")
    print(f"  humor_rating mean={rating_df['humor_rating'].mean():.2f}, std={rating_df['humor_rating'].std():.2f}")


if __name__ == "__main__":
    main()
