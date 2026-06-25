"""
Prepare NYCC (New Yorker Caption Contest) data for LoRA training and evaluation.

Source: datasets/newyorker/fold0_{train,test,validation}.csv
  - Each row is a pairwise caption contest: caption_a vs caption_b, expected winner (A/B)
  - image_description provides text context for the cartoon

Training set:
  - Sample 4k pairs from fold0_train → convert to winner/loser rows (8k rows total)
  - winner = 1.0, loser = 0.0
  - Output: datasets/lora_train_data/nycc.csv

Test set (pairwise benchmark):
  - fold0_test + fold0_validation (~1020 pairs)
  - Output: datasets/nycc_pairwise_test.csv
"""

import csv
import random
from pathlib import Path

SEED        = 42
TRAIN_PAIRS = 4000  # pairs from fold0_train → 8k rows total

ROOT     = Path(__file__).parent.parent
NY_DIR   = ROOT / "datasets" / "newyorker"
OUT_DIR  = ROOT / "datasets" / "lora_train_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUT = ROOT / "datasets" / "nycc_pairwise_test.csv"

random.seed(SEED)


def build_prompt(caption: str, image_desc: str) -> str:
    return (
        f"Consider the amount of funniness in the following New Yorker cartoon caption.\n\n"
        f"Image: {image_desc.strip()}\n"
        f"Caption: {caption.strip()}"
    )


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Training data ─────────────────────────────────────────────────────────────

train_rows = read_csv(NY_DIR / "fold0_train.csv")
random.shuffle(train_rows)
sampled = train_rows[:TRAIN_PAIRS]

out_rows = []
for row in sampled:
    img    = row["image_description"]
    winner = row["expected"]  # "A" or "B"
    cap_a  = row["caption_a"]
    cap_b  = row["caption_b"]

    winner_cap = cap_a if winner == "A" else cap_b
    loser_cap  = cap_b if winner == "A" else cap_a

    out_rows.append({"prompt_text": build_prompt(winner_cap, img), "score": 1.0, "source": "nycc"})
    out_rows.append({"prompt_text": build_prompt(loser_cap,  img), "score": 0.0, "source": "nycc"})

train_out = OUT_DIR / "nycc.csv"
with open(train_out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["prompt_text", "score", "source"])
    w.writeheader()
    w.writerows(out_rows)

print(f"Training: {len(out_rows)} rows → {train_out}")


# ── Test set (pairwise benchmark) ─────────────────────────────────────────────

test_rows = read_csv(NY_DIR / "fold0_test.csv")
val_rows  = read_csv(NY_DIR / "fold0_validation.csv")
all_test  = test_rows + val_rows
print(f"Test pairs: {len(all_test)} (fold0_test={len(test_rows)}, fold0_val={len(val_rows)})")

with open(TEST_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["caption_a", "caption_b", "expected", "image_description"])
    w.writeheader()
    for row in all_test:
        w.writerow({
            "caption_a":         row["caption_a"],
            "caption_b":         row["caption_b"],
            "expected":          row["expected"],
            "image_description": row["image_description"],
        })

print(f"Test benchmark: {len(all_test)} pairs → {TEST_OUT}")
