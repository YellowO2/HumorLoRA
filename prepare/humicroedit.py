"""
Prepare humicroedit + FunLines for joint pairwise training.

Combines train.csv (9,652) and train_funlines.csv (8,248), deduplicates by id.
Output: lora_train_data/humicroedit.csv with columns: prompt_text, score, source

Prompt format:
  Consider the amount of funniness in the following edited news headline.

  Edited:   Donald Trump misunderstands G7 talks on climate crisis and Amazon fires
  Original: Donald Trump ___ G7 talks on climate crisis and Amazon fires
"""
import re
import pandas as pd
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "datasets" / "humicroedit" / "semeval-2020-task-7-dataset" / "subtask-1"
OUT_PATH  = Path(__file__).parent.parent / "datasets" / "lora_train_data" / "humicroedit.csv"


def build_prompt(original: str, edit: str) -> str:
    filled = re.sub(r"<[^>]+/>", edit, original)
    blanked = re.sub(r"<[^>]+/>", "___", original)
    return (
        f"Consider the amount of funniness in the following edited news headline.\n\n"
        f"Edited:   {filled}\n"
        f"Original: {blanked}"
    )


def main():
    dfs = []
    for fname in ("train.csv", "train_funlines.csv"):
        path = DATA_DIR / fname
        if path.exists():
            dfs.append(pd.read_csv(path))
            print(f"Loaded {fname}: {len(dfs[-1])} rows")

    df = pd.concat(dfs).drop_duplicates(subset="id").reset_index(drop=True)
    before = len(df)
    df = df.dropna(subset=["original", "edit", "meanGrade"]).reset_index(drop=True)
    print(f"Combined (deduped): {before} rows → {len(df)} after dropping NaN")

    df["prompt_text"] = df.apply(lambda r: build_prompt(r["original"], r["edit"]), axis=1)
    df["source"] = "humicroedit"

    out = df[["prompt_text", "meanGrade", "source"]].rename(columns={"meanGrade": "score"})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\nOutput: {OUT_PATH}")
    print(f"Rows: {len(out)}, score range: {out['score'].min():.2f}–{out['score'].max():.2f}")
    print("\nSample prompt:")
    print(out["prompt_text"].iloc[0])


if __name__ == "__main__":
    main()
