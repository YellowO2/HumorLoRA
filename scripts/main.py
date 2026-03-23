from interact import ask_llm
from result_io import save_results
from dataset_helpers import load_examples
from prompt_helpers import build_prompt

# === Configuration ===

THINK = False
MODELS = ["qwen3:8b", "qwen3:14b", "qwen3:32b"] # available models: "qwen3:14b", "qwen3:4b-instruct", "qwen3:8b", "qwen3:32b", "qwen3:1.7b"
OUTPUT_INSTRUCTION = "The funnier caption is <A> or <B>?" # For jester, use the from_description as the prompt instead of the standard one. Set to empty string for default behavior.

# What to run
DATASETS = ["newyorker_default"]  # e.g., ["jester"], ["newyorker"], or both
NUM_RUNS = 2           # Number of times to repeat each evaluation

# Jester options
JESTER_DATASET_NUMS = [1, 2] # Runs these jester files if "jester" is in DATASETS
JESTER_OFFSET_DIVISOR = 2

# New Yorker options
NYCC_SPLIT = "validation"  # train / validation / test
NYCC_FOLDS = [0, 1, 2, 3]           # 0..4 where 0 corresponds to "ranking"
NYCC_LIMIT = 400           # e.g., 10 for quick test


# === Core Evaluation Logic ===
def run_evaluation(model_name: str, dataset_type: str, jester_num: int, nycc_fold: int):
    examples = load_examples(
        dataset=dataset_type,
        jester_dataset_num=jester_num,
        jester_offset_divisor=JESTER_OFFSET_DIVISOR,
        nycc_split=NYCC_SPLIT,
        nycc_fold=nycc_fold,
        nycc_limit=NYCC_LIMIT,
    )

    dataset_name_display = f"Jester DS{jester_num}" if dataset_type == "jester" else f"{dataset_type} Fold{nycc_fold}"
    print(f"\n--- Starting evaluation for {dataset_name_display} with model {model_name} ({len(examples)} comparisons) ---")

    results = []
    for i, sample in enumerate(examples):
        prompt = build_prompt(sample, OUTPUT_INSTRUCTION, default=True)
        expected = sample["expected"]

        try:
            llm_output = ask_llm(prompt, model=model_name, think=THINK)
            response = llm_output["content"]
            thinking = llm_output["thinking"]

            # Simple parsing for XML tag or fallback
            resp_up = response.strip()
            if "<A>" in resp_up or resp_up.endswith("A") or resp_up == "A":
                prediction = "A"
            elif "<B>" in resp_up or resp_up.endswith("B") or resp_up == "B":
                prediction = "B"
            else:
                prediction = "UNKNOWN"

            is_correct = (prediction == expected.upper())

            status = "✓" if is_correct else "✗"
            print(f"[{i + 1}] {sample['meta']}: LLM={prediction} (Raw={response!r}) Expected={expected} {status}")

            results.append(
                {
                    "dataset": dataset_type,
                    "sample_id": sample["sample_id"],
                    "meta": sample["meta"],
                    "expected": expected,
                    "llm_response": response,
                    "llm_thinking": thinking,
                    "is_correct": is_correct,
                }
            )
        except Exception as e:
            print(f"Error at sample {i+1}: {e}")
            break

    # Save results
    dataset_name_file = f"{dataset_type}_{NYCC_SPLIT}_fold{nycc_fold}" if "newyorker" in dataset_type else f"{dataset_type}_{jester_num}"

    save_results(
        results=results, 
        model_name=model_name,
        dataset_name=dataset_name_file,
        dataset_type=dataset_type,
        is_thinking=THINK,
        output_instruction=OUTPUT_INSTRUCTION,
    )


# === Main Execution Loop ===
if __name__ == "__main__":
    for run_idx in range(NUM_RUNS):
        print(f"\n==============================")
        print(f"       RUN {run_idx + 1} OF {NUM_RUNS}")
        print(f"==============================")
        
        for model in MODELS:
            if "jester" in DATASETS:
                for j_num in JESTER_DATASET_NUMS:
                    run_evaluation(model, "jester", jester_num=j_num, nycc_fold=0)
                    
            if "newyorker" in DATASETS or "newyorker_default" in DATASETS:
                for fold in NYCC_FOLDS:
                    ds_type = "newyorker_default" if "newyorker_default" in DATASETS else "newyorker"
                    run_evaluation(model, ds_type, jester_num=1, nycc_fold=fold)

