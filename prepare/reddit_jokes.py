"""
Clean raw Reddit jokes data and produce datasets/reddit_jokes/jokes.csv
for regression training, using upvote_ratio as the continuous label.

Filters applied:
  - score >= MIN_SCORE (minimum engagement, avoids noise from brand-new posts)
  - 20 <= char_len <= MAX_CHARS (drop trivially short / wall-of-text jokes)
  - Remove [removed] / [deleted] content
  - Exact dedup on lowercased text

Label: upvote_ratio (0–1), already normalized.
Higher ratio = more people found it good; combined with score filter it's a
reasonable crowd preference signal.

Run: python prepare/reddit_jokes.py
"""
import re
from pathlib import Path

import pandas as pd

MIN_SCORE = 10
MAX_CHARS = 2000
SEED      = 42

RAW_CSV  = Path(__file__).parent.parent / "datasets" / "reddit_jokes" / "reddit_full_data.csv"
OUT_CSV  = Path(__file__).parent.parent / "datasets" / "lora_train_data" / "reddit_jokes.csv"


def clean_text(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **bold**
    text = re.sub(r"\*(.+?)\*",     r"\1", text)    # *italic*
    text = re.sub(r"~~(.+?)~~",     r"\1", text)    # ~~strike~~
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url)
    text = re.sub(r"https?://\S+",  "",     text)   # bare URLs
    text = re.sub(r"\n{3,}",        "\n\n", text)   # collapse blank lines
    return text.strip()


def main():
    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    print(f"  Raw rows: {len(df)}")

    # Drop [removed] / [deleted]
    mask_bad = df["fulltext"].str.contains(r"\[removed\]|\[deleted\]", regex=True, na=False)
    df = df[~mask_bad]
    print(f"  After dropping [removed]/[deleted]: {len(df)}")

    # Minimum engagement filter
    df = df[df["score"] >= MIN_SCORE].copy()
    print(f"  After score >= {MIN_SCORE}: {len(df)}")

    # Length filter
    df = df[(df["char_len"] >= 20) & (df["char_len"] <= MAX_CHARS)].copy()
    print(f"  After length filter [20, {MAX_CHARS}]: {len(df)}")

    # Clean text
    df["text"] = df["fulltext"].apply(clean_text)

    # Drop empties after cleaning
    df = df[df["text"].str.len() > 0]

    # Exact dedup on normalized text
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"  After dedup: {len(df)}  (dropped {before - len(df)} exact dupes)")

    # Final output — unified schema for lora_train_data/
    out = df.reset_index(drop=True)
    out["prompt_text"] = out["text"].apply(
        lambda t: f"Consider the amount of funniness in the following: {t}"
    )
    out["source"] = "reddit_jokes"
    out = out[["prompt_text", "upvote_ratio", "source"]].rename(columns={"upvote_ratio": "score"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(out)} rows to {OUT_CSV}")
    print(f"  score (upvote_ratio): mean={out['score'].mean():.3f}  "
          f"min={out['score'].min():.2f}  max={out['score'].max():.2f}")


if __name__ == "__main__":
    main()
