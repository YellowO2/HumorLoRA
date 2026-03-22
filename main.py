import pandas as pd
import random
from interact import ask_llm
from result_io import build_output_file, save_accuracy_summary, save_results

# Load and prepare data
df = pd.read_csv("datasets/cleaned_ranked_dataset3.csv")
df = df.dropna(subset=['avg_rating', 'joke_text'])

# Parameters
OFFSET = len(df) // 2  # Compare jokes that are this far apart in rank to ensure a meaningful difference
THINK = True
MODEL = "qwen3:14b"
results = []

OUTPUT_FILE = build_output_file(MODEL, is_thinking=THINK)
print(f"Starting evaluation with {len(df) - OFFSET} comparisons...\n")

# Run comparisons
for i in range(len(df) - OFFSET):
    joke_better = df.iloc[i]
    joke_worse = df.iloc[i + OFFSET]
    
    # Randomize position to avoid position bias
    if random.choice([True, False]):
        joke_a, joke_b = joke_better, joke_worse
        expected = "Joke A"
    else:
        joke_a, joke_b = joke_worse, joke_better
        expected = "Joke B"
    
    prompt = f"""Which joke is funnier?

Joke A:
{joke_a['joke_text']}

Joke B:
{joke_b['joke_text']}

Answer with only "Joke A" or "Joke B"."""
    
    try:
        llm_output = ask_llm(prompt, model=MODEL, think=THINK)
        response = llm_output["content"]
        thinking = llm_output["thinking"]
        response_clean = response.replace(".", "").replace('"', '').strip()
        is_correct = response_clean.lower() == expected.lower()
        
        print(f"[{i+1}] Rank {joke_better['rank']} vs {joke_worse['rank']}: LLM={response_clean} Expected={expected} ✓" if is_correct else f"[{i+1}] Rank {joke_better['rank']} vs {joke_worse['rank']}: LLM={response_clean} Expected={expected} ✗")
        
        results.append({
            "rank_better": joke_better['rank'],
            "rank_worse": joke_worse['rank'],
            "expected": expected,
            "llm_response": response,
            "llm_thinking": thinking,
            "is_correct": is_correct
        })
    except Exception as e:
        print(f"Error: {e}")
        break

# Save results
results_df = save_results(results, OUTPUT_FILE)

if not results_df.empty:
    accuracy = results_df['is_correct'].mean() * 100
    summary_file = save_accuracy_summary(
        model_name=MODEL,
        output_file=OUTPUT_FILE,
        is_thinking=THINK,
        accuracy_percent=accuracy,
        correct_count=int(results_df['is_correct'].sum()),
        total_count=len(results_df),
    )
    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Accuracy summary updated: {summary_file}")

