"""
Prepare HAHA Spanish 2019 for joint pairwise training.

Source: datasets/haha_spanish/train.csv (24,000 rows)
- Humorous (is_humor=1): use funniness_average as score (1.0–5.0)
- Non-humorous (is_humor=0): score = 0 (annotators voted ~3.5/5 "not humorous"; legitimate floor)

Output: lora_train_data/haha_spanish.csv with columns: prompt_text, score, source

Prompt format:
  Consider the amount of funniness in the following Spanish tweet: {text}
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "datasets" / "haha_spanish"
OUT_PATH = Path(__file__).parent.parent / "datasets" / "lora_train_data" / "haha_spanish.csv"


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"Loaded: {len(df)} rows")

    humor = df[df["is_humor"] == 1].copy()
    non_humor = df[df["is_humor"] == 0].copy()
    print(f"  Humorous: {len(humor)}, Non-humorous: {len(non_humor)}")

    humor["score"] = humor["funniness_average"]
    non_humor["score"] = 0.0

    out = pd.concat([humor, non_humor])[["id", "text", "score"]].copy()
    out["prompt_text"] = out["text"].apply(
        lambda t: f"Consider the amount of funniness in the following Spanish tweet: {t}"
    )
    out["source"] = "haha_spanish"
    out = out[["prompt_text", "score", "source"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\nOutput: {OUT_PATH}")
    print(f"Rows: {len(out)}, score range: {out['score'].min():.2f}–{out['score'].max():.2f}")
    print(f"Score distribution:\n{out['score'].value_counts().sort_index().head(10)}")
    print("\nSample prompt (humorous):")
    print(out[out["score"] > 0]["prompt_text"].iloc[0])


if __name__ == "__main__":
    main()
