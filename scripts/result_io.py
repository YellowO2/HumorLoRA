import os
from pathlib import Path
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sanitize_model_name(model_name: str) -> str:
    return (
        model_name.strip()
        .replace(":", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
    )


def build_output_file(model_name: str, output_dir: str | Path, is_thinking: bool = False) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_model = sanitize_model_name(model_name)
    suffix = "_thinking" if is_thinking else ""
    prefix = f"llm_eval_{safe_model}{suffix}_"

    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".csv", dir=out_dir)
    os.close(fd)  # close descriptor; pandas will write later
    return Path(path)


def save_results(
    results: list[dict],
    model_name: str,
    dataset_name: str,
    dataset_type: str,
    is_thinking: bool = False,
    output_instruction: str = "",
) -> pd.DataFrame:
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        return results_df
        
    output_dir = PROJECT_ROOT / "results" / dataset_type
    output_file = build_output_file(model_name, output_dir=output_dir, is_thinking=is_thinking)
    results_df.to_csv(output_file, index=False)
    
    save_accuracy_summary(
        results_df=results_df,
        model_name=model_name,
        output_file=output_file,
        dataset_name=dataset_name,
        summary_file=output_dir / "accuracy_summary.csv",
        is_thinking=is_thinking,
        output_instruction=output_instruction,
    )
    
    return results_df


def save_accuracy_summary(
    results_df: pd.DataFrame,
    *,
    model_name: str,
    output_file: Path,
    dataset_name: str,
    summary_file: str | Path,
    is_thinking: bool = False,
    output_instruction: str = ""
) -> Path:
    summary_path = Path(summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    correct_count = int(results_df["is_correct"].sum())
    total_count = len(results_df)
    accuracy_percent = (correct_count / total_count) * 100 if total_count > 0 else 0.0

    row = {
        "model": f"{model_name}{' (thinking)' if is_thinking else ''}",
        "output_file": output_file.name,
        "dataset": dataset_name,
        "accuracy_percent": round(accuracy_percent, 3),
        "correct": correct_count,
        "total": total_count,
        "output_instruction": output_instruction,
    }

    if summary_path.exists():
        pd.DataFrame([row]).to_csv(summary_path, mode="a", header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(summary_path, index=False)

    print(f"\n✓ Results saved to {output_file}")
    print(f"Accuracy: {accuracy_percent:.1f}% ({correct_count}/{total_count})")
    print(f"Accuracy summary updated: {summary_path}")

    return summary_path