"""
Download HaHackathon train split and produce two datasets with a proper 80/20 split:
  datasets/hahackathon/rating.csv    — 80% of humorous jokes for LoRA training
  datasets/hahackathon/pairwise.csv  — pairs formed exclusively from held-out 20%

No gap filter on pairwise — unfiltered random pairs give an honest eval number.
"""
import random
from pathlib import Path

import pandas as pd

TRAIN_URL = (
    "https://raw.githubusercontent.com/NLP-UMUTeam/SemEval2021-HaHackathon-UMUTeam"
    "/main/datasets/hahackathon_train.csv"
)

TRAIN_FRAC = 0.8
N_PAIRS    = 2000
SEED       = 42

OUT_DIR      = Path(__file__).parent.parent / "datasets" / "hahackathon"
TRAIN_OUT    = Path(__file__).parent.parent / "datasets" / "lora_train_data" / "hahackathon.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("Downloading HaHackathon train split...")
    df = pd.read_csv(TRAIN_URL)
    print(f"Total rows: {len(df)}")

    humorous = df[df["is_humor"] == 1].sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Humorous texts: {len(humorous)}")

    # ── 80/20 split by joke ID — no overlap ───────────────────────────────────
    n_train = int(len(humorous) * TRAIN_FRAC)
    train_df = humorous.iloc[:n_train]
    test_df  = humorous.iloc[n_train:]
    print(f"Train: {len(train_df)} jokes  |  Test (held-out): {len(test_df)} jokes")

    # ── Rating dataset (train split) — unified schema for lora_train_data/ ──────
    rating_df = train_df[["id", "text", "humor_rating"]].reset_index(drop=True)
    lora_df = rating_df.copy()
    lora_df["prompt_text"] = lora_df["text"].apply(
        lambda t: f"Consider the amount of funniness in the following: {t}"
    )
    lora_df["source"] = "hahackathon"
    lora_df = lora_df[["prompt_text", "humor_rating", "source"]].rename(columns={"humor_rating": "score"})
    lora_df.to_csv(TRAIN_OUT, index=False)
    print(f"\nLoRA train: {len(lora_df)} rows saved to {TRAIN_OUT}")
    print(f"  humor_rating mean={lora_df['score'].mean():.2f}, std={lora_df['score'].std():.2f}")

    # ── Pairwise dataset (held-out test split only, no gap filter) ────────────
    rng  = random.Random(SEED)
    rows = test_df.to_dict("records")
    indices = list(range(len(rows)))

    pairs = []
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < N_PAIRS * 100:
        i, j = rng.sample(indices, 2)
        a, b = rows[i], rows[j]
        if a["humor_rating"] == b["humor_rating"]:
            attempts += 1
            continue
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
    print(f"\nPairwise: {len(pairwise_df)} pairs saved to {pairwise_path}")
    gap = (pairwise_df["rating_a"] - pairwise_df["rating_b"]).abs()
    print(f"  avg gap: {gap.mean():.2f}  |  min: {gap.min():.2f}  |  max: {gap.max():.2f}")
    print(f"  expected A: {(pairwise_df['expected']=='A').sum()}  B: {(pairwise_df['expected']=='B').sum()}")


if __name__ == "__main__":
    main()
