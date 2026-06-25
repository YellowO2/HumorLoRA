"""
Prepare Humor Arena pairwise data for joint LoRA training.

Source: datasets/humor_arena/humor_arena.choices.20250812.json
  2,541 entries: LEFT=594, RIGHT=619, NONE=1186, BOTH=142
  NONE/BOTH (ties) are dropped — 1,213 usable comparisons.

Each comparison expands to 2 rows (winner score=1.0, loser score=0.0).
Dedup on prompt_text since the same joke can appear in multiple comparisons.

Output: lora_train_data/humor_arena.csv with columns: prompt_text, score, source
"""
import json
import pandas as pd
from pathlib import Path

SRC  = Path(__file__).parent.parent / "datasets" / "humor_arena" / "humor_arena.choices.20250812.json"
OUT  = Path(__file__).parent.parent / "datasets" / "lora_train_data" / "humor_arena.csv"

PREFIX = "Consider the amount of funniness in the following: "


def main():
    with open(SRC) as f:
        raw = json.load(f)
    print(f"Loaded: {len(raw)} entries")

    rows = []
    skipped = 0
    for entry in raw:
        winner = entry["winner"]
        if winner in ("NONE", "BOTH"):
            skipped += 1
            continue
        left  = entry["left_joke"]
        right = entry["right_joke"]
        if winner == "LEFT":
            rows.append({"prompt_text": PREFIX + left,  "score": 1.0})
            rows.append({"prompt_text": PREFIX + right, "score": 0.0})
        else:
            rows.append({"prompt_text": PREFIX + left,  "score": 0.0})
            rows.append({"prompt_text": PREFIX + right, "score": 1.0})

    print(f"Skipped ties (NONE/BOTH): {skipped}")

    df = pd.DataFrame(rows).drop_duplicates(subset="prompt_text").reset_index(drop=True)
    df["source"] = "humor_arena"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\nOutput: {OUT}")
    print(f"Rows: {len(df)}, score range: {df['score'].min():.1f}–{df['score'].max():.1f}")
    print("\nSample:")
    print(df["prompt_text"].iloc[0][:120])


if __name__ == "__main__":
    main()
