from pathlib import Path
import os
import pandas as pd
from datasets import load_dataset, load_from_disk


OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "newyorker"
RAW_DIR = OUTPUT_DIR / "raw"
PREPARED_DIR = OUTPUT_DIR
TASK = "ranking"
FOLDS = [0, 1, 2, 3, 4] # exist in folds, meaning just different combinations of train and test
SPLITS = ["train", "validation", "test"]
HF_TOKEN = os.getenv("HF_TOKEN")


def config_name_from_fold(fold: int) -> str:
    return "ranking" if fold == 0 else f"ranking_{fold}"


def normalize_row(row: dict, split: str) -> dict:
    choices = row["caption_choices"]
    label = row["label"]

    return {
        "sample_id": row["instance_id"],
        "meta": f"NYCC {split} {row['instance_id']}",
        "joke_a_text": choices[0],
        "joke_b_text": choices[1],
        "expected": "A" if label == "A" else "B",
        "winner_source": row["winner_source"],
        "image_description": row.get("image_description"),
        "image_uncanny_description" : row.get("image_uncanny_description", ""),
        "from_description": row.get("from_description", ""),
    }


def export_fold(fold: int) -> None:
    cfg = config_name_from_fold(fold)
    raw_fold_dir = RAW_DIR / cfg

    if raw_fold_dir.exists():
        print(f"Loading {TASK} fold {fold} ({cfg}) from local raw snapshot...")
        dset = load_from_disk(str(raw_fold_dir))
    else:
        print(f"Downloading {TASK} fold {fold} ({cfg}) from Hugging Face...")
        dset = load_dataset(
            "jmhessel/newyorker_caption_contest",
            cfg,
            token=HF_TOKEN,
        )
        raw_fold_dir.parent.mkdir(parents=True, exist_ok=True)
        dset.save_to_disk(str(raw_fold_dir))
        print(f"  - Saved raw snapshot to {raw_fold_dir}")

    for split in SPLITS:
        split_data = dset[split]
        rows = [normalize_row(row, split=split) for row in split_data]
        out_path = PREPARED_DIR / f"ranking_fold{fold}_{split}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  - Saved {len(rows)} rows to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not HF_TOKEN:
        print("Token not set. Set token with: export HF_TOKEN='your_token'")
    for fold in FOLDS:
        export_fold(fold)
    print("Done. In datasets/newyorker/")


if __name__ == "__main__":
    main()
