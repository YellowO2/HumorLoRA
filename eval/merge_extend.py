"""
Merge two partial result CSVs (e.g. examples 0-999 and 1000-1999) into one
and print combined accuracy. Also appends a row to summary.csv.

Usage:
    python eval/merge_extend.py results/A.csv results/B.csv [--label my-model-no-gut]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--label", required=True, help="Model label for summary.csv")
    parser.add_argument("--dataset", default="nycc")
    parser.add_argument("--out", default=None, help="Output CSV path (optional)")
    args = parser.parse_args()

    df_a = pd.read_csv(args.file_a)
    df_b = pd.read_csv(args.file_b)

    # Sanity check: no overlapping sample_ids
    overlap = set(df_a["sample_id"]) & set(df_b["sample_id"])
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping sample_ids — check your offsets!", file=sys.stderr)

    combined = pd.concat([df_a, df_b], ignore_index=True)
    acc = combined["is_correct"].mean() * 100
    n = len(combined)
    unknowns = int((combined["prediction"] == "UNKNOWN").sum())

    print(f"Combined: n={n}, accuracy={acc:.1f}%, unknowns={unknowns}")

    out_path = args.out or str(Path(args.file_a).parent / f"{args.label}_merged.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    from datetime import datetime
    row = {
        "model": args.label,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_examples": n,
        "overall_acc": round(acc, 1),
        "unknown_count": unknowns,
        "dataset": args.dataset,
    }
    summary_df = pd.DataFrame([row])
    if SUMMARY_PATH.exists():
        summary_df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary updated: {row}")


if __name__ == "__main__":
    main()
