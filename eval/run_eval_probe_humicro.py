# ═══════════════════════════════════════════════════════════════════════════════
# Activation Probe: finding Qwen's internal "funniness direction"
# using Humicroedit (SemEval-2020 Task 7)
# ═══════════════════════════════════════════════════════════════════════════════
#
# KEY IDEA (Representation Engineering / RepE):
#
#   When Qwen processes text, each transformer layer produces a hidden state
#   vector for every token — a point in high-dimensional space (dim ~3072).
#   If Qwen internally "understands" funniness, then funny headlines and
#   unfunny headlines should land in slightly different regions of that space.
#
#   We find this "funniness direction" by:
#     1. Collecting hidden states for ~1000 funny headlines and ~800 unfunny ones
#     2. Fitting a logistic regression: can a linear boundary separate them?
#     3. The weight vector of that classifier IS the funniness direction
#     4. To score a new text: project its hidden state onto that direction
#
#   Why Humicroedit is ideal for this:
#     - Same structure (news headline), only ONE word differs between funny/unfunny
#     - Controls for length, topic, format — only funniness varies
#     - meanGrade >= 2.0 = funny (clear signal), meanGrade == 0.0 = not funny at all
#
#   If probe accuracy >> 50%:  Qwen already encodes funniness linearly
#   If probe accuracy ~ 50%:   No linear signal — funniness is not a simple direction
#
# Run: python eval/run_eval_probe_humicro.py
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PROBE_LAYERS  = [8, 12, 16, 20, 24]
BASE_MODEL    = "unsloth/Qwen3.5-4B"
MAX_LENGTH    = 256
BATCH_SIZE    = 8
SEED          = 42

DATA_PATH    = Path(__file__).parent.parent / "datasets" / "humicroedit" / "probe_data.csv"
# Held-out eval: HaHa pairwise (different domain — standalone jokes not headlines)
HAHA_PATH    = Path(__file__).parent.parent / "datasets" / "reward" / "test_haha.csv"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "probe"
SUMMARY_PATH = Path(__file__).parent.parent / "results" / "summary.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"    # causal LM: last token = most context-aware

def make_prompt(text: str) -> str:
    # RepE-style prompt: explicitly frames the task as evaluating the concept
    # (Zou et al. 2023 template: "Consider the amount of <emotion> in the following: <stimulus>")
    # This activates funniness-relevant circuits rather than triggering a generic response.
    # add_generation_prompt=True puts model in "about to respond" state — last token
    # hidden states best reflect what the model "thinks" about the input at this point.
    messages = [{"role": "user", "content": f"Consider the amount of funniness in the following: {text}"}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

print("Loading Qwen in 4-bit (no reward head — plain language model)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    output_hidden_states=True,    # expose all layer outputs
)
model.eval()
print(f"Model ready. Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}\n")


# ── Extract hidden states ─────────────────────────────────────────────────────
# For each text, we collect the hidden state vector at the LAST token position
# from multiple layers. The last token is most context-aware in a causal LM.

@torch.inference_mode()
def get_hidden_states(texts: list[str], layers: list[int]) -> dict[int, np.ndarray]:
    """
    Returns {layer_idx: array of shape (N, hidden_size)}.
    Each row is the hidden state at the last token for one input text.
    """
    layer_acts = {l: [] for l in layers}

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        prompts = [make_prompt(t) for t in batch]
        enc = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        out = model(**enc, output_hidden_states=True)
        # hidden_states[0] = embedding layer, hidden_states[i] = layer i output
        for l in layers:
            h = out.hidden_states[l][:, -1, :]    # (batch, hidden_size) — last token
            layer_acts[l].append(h.float().cpu().numpy())

        if start % (BATCH_SIZE * 10) == 0:
            print(f"  {min(start + BATCH_SIZE, len(texts))}/{len(texts)} texts")

    return {l: np.concatenate(layer_acts[l], axis=0) for l in layers}


# ── Load Humicroedit data ─────────────────────────────────────────────────────

print("Loading Humicroedit probe data...")
df = pd.read_csv(DATA_PATH)
print(f"  Funny (label=1): {(df['label']==1).sum()}")
print(f"  Unfunny (label=0): {(df['label']==0).sum()}\n")

texts  = df["text"].tolist()
labels = df["label"].values

print(f"Extracting hidden states for {len(texts)} Humicroedit examples...")
acts = get_hidden_states(texts, PROBE_LAYERS)


# ── Fit and cross-validate probes ─────────────────────────────────────────────
# We fit a logistic regression at each layer.
# Cross-validation (5-fold) gives honest accuracy without a separate val set.
# The weight vector of the fitted classifier = the "funniness direction".

print("\nFitting linear probes per layer (5-fold cross-validation)...")
probes  = {}
scalers = {}
cv_accs = {}

for l in PROBE_LAYERS:
    X = acts[l]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    cv_scores = cross_val_score(clf, X_s, labels, cv=5, scoring="accuracy")
    cv_accs[l] = cv_scores.mean() * 100

    # fit on full data for use in downstream eval
    clf.fit(X_s, labels)
    probes[l]  = clf
    scalers[l] = scaler

    print(f"  Layer {l:2d}: {cv_accs[l]:.1f}% ± {cv_scores.std()*100:.1f}%")

best_layer = max(cv_accs, key=cv_accs.get)
print(f"\nBest layer: {best_layer} ({cv_accs[best_layer]:.1f}%)")


# ── Cross-domain eval: HaHa pairwise ──────────────────────────────────────────
# Probe trained on Humicroedit headlines → tested on standalone jokes.
# For each pair, score both jokes and check if score(chosen) > score(rejected).
# This tests whether the funniness direction generalises across text types.

print("\nCross-domain eval: HaHa pairwise (probe trained on headlines, tested on jokes)...")
haha = pd.read_csv(HAHA_PATH)
print(f"Loaded {len(haha)} held-out HaHa pairs\n")

all_texts = haha["chosen"].tolist() + haha["rejected"].tolist()
print(f"Extracting hidden states for {len(all_texts)} HaHa texts...")
haha_acts = get_hidden_states(all_texts, PROBE_LAYERS)

n = len(haha)
haha_accs = {}
for l in PROBE_LAYERS:
    X = haha_acts[l]
    X_s = scalers[l].transform(X)
    scores = probes[l].decision_function(X_s)   # higher = more likely funny
    scores_chosen   = scores[:n]
    scores_rejected = scores[n:]
    acc = (scores_chosen > scores_rejected).mean() * 100
    haha_accs[l] = acc

print("\n══ Results ══")
print(f"{'Layer':>6}  {'Humicro CV':>11}  {'HaHa xdomain':>13}")
for l in PROBE_LAYERS:
    print(f"  {l:4d}  {cv_accs[l]:10.1f}%  {haha_accs[l]:12.1f}%")

print(f"\nBest layer {best_layer}: Humicro {cv_accs[best_layer]:.1f}%  |  HaHa cross-domain {haha_accs[best_layer]:.1f}%")
print()
print("Interpretation:")
print(f"  Humicro CV >> 50%  →  Qwen has a linear funniness direction in its activations")
print(f"  HaHa xdomain >> 50%  →  that direction generalises to standalone jokes")
print(f"  Both ~ 50%  →  no simple linear humor signal; need different approach")


# ── Save results ──────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df = pd.DataFrame([
    {
        "layer":       l,
        "humicro_cv":  round(cv_accs[l], 1),
        "haha_xdomain": round(haha_accs[l], 1),
    }
    for l in PROBE_LAYERS
])
out_path = RESULTS_DIR / f"probe_humicro_{timestamp}.csv"
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

row = {
    "model":       "qwen4b-probe-humicro",
    "timestamp":   timestamp,
    "n_examples":  len(haha),
    "overall_acc": round(haha_accs[best_layer], 1),
    "unknown_count": 0,
    "dataset":     f"haha-probe-xdomain-layer{best_layer}",
}
summary_df = pd.DataFrame([row])
if SUMMARY_PATH.exists():
    summary_df.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)
else:
    summary_df.to_csv(SUMMARY_PATH, index=False)
print("Summary updated.")
