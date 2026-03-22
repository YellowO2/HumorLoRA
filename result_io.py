import os
from pathlib import Path
import tempfile

import pandas as pd


def sanitize_model_name(model_name: str) -> str:
    return (
        model_name.strip()
        .replace(":", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
    )


def build_output_file(model_name: str, output_dir: str = "results", is_thinking: bool = False) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    safe_model = sanitize_model_name(model_name)
    suffix = "_thinking" if is_thinking else ""
    prefix = f"llm_eval_{safe_model}{suffix}_"

    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".csv", dir=output_dir)
    os.close(fd)  # close descriptor; pandas will write later
    return Path(path)


def save_results(results: list[dict], output_file: Path) -> pd.DataFrame:
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    return results_df


def save_accuracy_summary(
    *,
    model_name: str,
    is_thinking: bool,
    output_file: Path,
    accuracy_percent: float,
    correct_count: int,
    total_count: int,
    summary_file: str = "results/accuracy_summary.csv",
) -> Path:
    summary_path = Path(summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "model": model_name + (" (thinking)" if is_thinking else ""),
        "output_file": output_file.name, #only the filename, not full path
        "accuracy_percent": round(accuracy_percent, 3),
        "correct": correct_count,
        "total": total_count,
    }

    if summary_path.exists():
        pd.DataFrame([row]).to_csv(summary_path, mode="a", header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(summary_path, index=False)

    return summary_path